"""
diagnose/sft_score_trend.py

Quick trend diagnostic: compute mean log P(chosen_sid | history) and
mean log P(rejected_sid | history) on the v1 engagement valid set, across
multiple LoRA checkpoints. NO beam search, NO generate — just forward +
score the 3 SID tokens. ~3 min per model.

Why this exists:
  evaluate_engaged.py uses beam-search generate (1000 samples × ~1.75s =
  ~30 min per model). For a 6-model trend (base + 5 SFT ckpts) that's 3h.
  But the question we usually want answered is "did the model's P(chosen)
  go up across ckpts?" — which is what the SFT loss directly optimizes.
  This script answers it via direct forward, ~15 min for 6 models.

  generate-based Recall@K and direct log P(chosen) are highly correlated
  — that's the rationale for `--best_metric chosen_score` in the trainers.
  If log P(chosen) is flat across ckpts here, Recall@K will also be flat
  (within sampling noise). Conversely, this script catches subtle changes
  (0.05 nat shifts) that beam-search recall would average away.

Usage (from project root):
    python diagnose/sft_score_trend.py \\
        --base model/OneRec-1.7B \\
        --adapters runs/sft_only_5k/checkpoint-125 \\
                   runs/sft_only_5k/checkpoint-250 \\
                   runs/sft_only_5k/checkpoint-375 \\
                   runs/sft_only_5k/checkpoint-500 \\
                   runs/sft_only_5k/checkpoint-625 \\
        --include_base \\
        --valid_parquet data/contrastive_dataset_v1/valid.parquet \\
        --template model/qwen3_soft_switch.jinja2 \\
        --n 1000 \\
        --output_csv runs/sft_score_trend.csv

Output: trend table with rows per checkpoint, columns:
    chosen_score, rejected_score, Δ (= chosen - rejected),
    pref_acc (fraction of rows where chosen > rejected),
    n_chosen, n_rejected (sanity counts)
"""

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "train"))
from dataset import SYSTEM_PROMPT, build_history_text, sid_to_core_text  # noqa: E402
from utils import resolve_template  # noqa: E402

NUM_SID_TOKENS = 3


# ===========================================================================
# Dataset — flatten v1 valid into one example per (row, sid_idx, sid_type)
# ===========================================================================


class FlatSIDDataset(Dataset):
    """Yields one tokenized sequence per (row, sid_index, sid_type) triple.
    Each row of v1 valid has 3 chosen + 3 rejected SIDs, so this is 6× the
    number of input rows."""

    def __init__(self, valid_df, tokenizer, max_hist=512, max_total_len=3072):
        self.tokenizer = tokenizer
        self.max_hist = max_hist
        self.max_total_len = max_total_len

        # Pre-flatten all (row_idx, sid, sid_type) triples for indexing.
        items = []
        for row_idx in range(len(valid_df)):
            row = valid_df.iloc[row_idx]
            for sid_idx, sid in enumerate(row["chosen_sids"]):
                items.append((row_idx, sid_idx, "chosen", row["hist_sids"], sid))
            for sid_idx, sid in enumerate(row["rejected_sids"]):
                items.append((row_idx, sid_idx, "rejected", row["hist_sids"], sid))
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        row_idx, sid_idx, sid_type, hist_sids, sid = self.items[idx]

        hist_text = build_history_text(hist_sids, max_hist=self.max_hist)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": hist_text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) + "<|sid_begin|>"

        full_text = prompt + sid_to_core_text(sid)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=True)["input_ids"]
        prompt_len = len(prompt_ids)

        assert len(full_ids) == prompt_len + NUM_SID_TOKENS, (
            f"tokenizer drift: full={len(full_ids)} vs prompt+3={prompt_len + 3}"
        )

        if self.max_total_len is not None and len(full_ids) > self.max_total_len:
            cut = len(full_ids) - self.max_total_len
            full_ids = full_ids[cut:]
            prompt_len = prompt_len - cut

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "prompt_len": prompt_len,
            "row_idx": row_idx,
            "sid_idx": sid_idx,
            "sid_type": sid_type,  # "chosen" or "rejected"
        }


def flat_collate(batch, pad_token_id):
    seqs = [b["input_ids"] for b in batch]
    max_len = max(s.size(0) for s in seqs)
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    attn_mask = torch.zeros((B, max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, :s.size(0)] = s
        attn_mask[i, :s.size(0)] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "prompt_lens": torch.tensor([b["prompt_len"] for b in batch], dtype=torch.long),
        "row_idxs": [b["row_idx"] for b in batch],
        "sid_types": [b["sid_type"] for b in batch],
    }


# ===========================================================================
# Score
# ===========================================================================


def compute_sid_score(last_hidden, lm_head, input_ids, prompt_lens):
    """Mean log P(SID token | prefix) over the 3 SID tokens. Returns (B,).

    Caller must wrap the surrounding forward in `torch.inference_mode()` —
    otherwise the transformer forward retains all 28 layers of bf16
    activations (~60GB at batch=16, L=3072), which OOM'd on a 95GB H100.
    """
    B, _, H = last_hidden.shape
    prompt_lens = prompt_lens.to(last_hidden.device, non_blocking=True)
    ks = torch.arange(NUM_SID_TOKENS, device=last_hidden.device)
    pred_pos = prompt_lens.unsqueeze(1) - 1 + ks
    token_pos = prompt_lens.unsqueeze(1) + ks

    idx = pred_pos.unsqueeze(-1).expand(-1, -1, H)
    slice_hidden = last_hidden.gather(1, idx)
    slice_logits = lm_head(slice_hidden).float()         # (B, K, V) fp32

    target_ids = input_ids.gather(1, token_pos)
    token_logits = slice_logits.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    log_norm = torch.logsumexp(slice_logits, dim=-1)
    token_logp = token_logits - log_norm                 # (B, K)
    return token_logp.mean(dim=1)                        # (B,)


def evaluate_model(model, dataset, pad_token_id, batch_size, label, device):
    """Run forward pass over all 6N items, return per-(row, sid_type) scores
    averaged over the 3 SIDs."""
    causal_lm = (
        model.get_base_model() if hasattr(model, "get_base_model") else model
    )
    transformer = causal_lm.model
    lm_head = causal_lm.lm_head

    collate = lambda b: flat_collate(b, pad_token_id=pad_token_id)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate, drop_last=False)

    # Accumulate per (row_idx, sid_type) → list of 3 SID scores.
    chosen_scores_per_row = {}    # row_idx -> [score_0, score_1, score_2]
    rejected_scores_per_row = {}

    model.eval()
    for i, batch in enumerate(tqdm(loader, desc=f"score {label}")):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn_mask = batch["attention_mask"].to(device, non_blocking=True)
        prompt_lens = batch["prompt_lens"]

        # Critical: wrap BOTH transformer + lm_head in inference_mode so
        # activations are not retained for autograd. Without this, a 95GB
        # H100 OOMs at batch=16, L=3072 (~60GB of 28-layer bf16 activations).
        with torch.inference_mode():
            out = transformer(
                input_ids=input_ids, attention_mask=attn_mask, use_cache=False,
            )
            scores = compute_sid_score(
                out.last_hidden_state, lm_head, input_ids, prompt_lens,
            )
        scores_cpu = scores.float().cpu().tolist()
        del input_ids, attn_mask, out, scores

        for row_idx, sid_type, score in zip(batch["row_idxs"],
                                            batch["sid_types"],
                                            scores_cpu):
            d = chosen_scores_per_row if sid_type == "chosen" else rejected_scores_per_row
            d.setdefault(row_idx, []).append(score)

        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()

    # Each row should have exactly 3 chosen + 3 rejected scores.
    rows = []
    for row_idx in sorted(set(chosen_scores_per_row) | set(rejected_scores_per_row)):
        cs = chosen_scores_per_row.get(row_idx, [])
        rs = rejected_scores_per_row.get(row_idx, [])
        assert len(cs) == 3, f"row {row_idx}: chosen has {len(cs)} scores"
        assert len(rs) == 3, f"row {row_idx}: rejected has {len(rs)} scores"
        c_mean = sum(cs) / 3
        r_mean = sum(rs) / 3
        rows.append({
            "row_idx": row_idx,
            "chosen_score": c_mean,
            "rejected_score": r_mean,
            "delta": c_mean - r_mean,
            "chosen_gt_rejected": c_mean > r_mean,
        })
    return pd.DataFrame(rows)


def summarize(label, df, path=None):
    return {
        "checkpoint": label,
        "path": path or "",
        "n": len(df),
        "chosen_score": df["chosen_score"].mean(),
        "rejected_score": df["rejected_score"].mean(),
        "delta": df["delta"].mean(),
        "pref_acc": df["chosen_gt_rejected"].mean(),
    }


# ===========================================================================
# Main
# ===========================================================================


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--adapters", nargs="+", required=True)
    p.add_argument("--include_base", action="store_true")
    p.add_argument("--valid_parquet",
                   default="data/contrastive_dataset_v1/valid.parquet")
    p.add_argument("--template", default=None)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--max_hist", type=int, default=512)
    p.add_argument("--max_total_len", type=int, default=3072)
    p.add_argument(
        "--batch_size", type=int, default=16,
        help="Forward-only inference (no_grad, no LoRA backward, no grad "
             "checkpointing). 16 fits easily; bump to 32 if VRAM allows.",
    )
    p.add_argument("--subsample_seed", type=int, default=67890,
                   help="Match evaluate_engaged.py's default for direct "
                        "comparison to its Recall@K numbers.")
    p.add_argument("--output_csv", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    warnings.filterwarnings("ignore")
    hf_logging.set_verbosity_error()

    template_path = resolve_template(args.template)

    print(f"Loading valid set from {args.valid_parquet} ...")
    valid_df = pd.read_parquet(args.valid_parquet).reset_index(drop=True)
    n_full = len(valid_df)
    n = n_full if args.n < 0 else min(args.n, n_full)
    print(f"  {n_full:,} total rows, scoring {n}")
    if n < n_full:
        rng = torch.Generator().manual_seed(args.subsample_seed)
        idx = torch.randperm(n_full, generator=rng)[:n].tolist()
        valid_df = valid_df.iloc[idx].reset_index(drop=True)

    print(f"Loading tokenizer + base model from {args.base} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tokenizer.chat_template = template_path.read_text(encoding="utf-8")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
        attn_implementation=attn_impl,
    )
    base.eval()

    print("Building flat (row, sid_idx, sid_type) dataset ...")
    flat_ds = FlatSIDDataset(valid_df, tokenizer,
                             max_hist=args.max_hist,
                             max_total_len=args.max_total_len)
    print(f"  {len(flat_ds):,} sequences total ({len(valid_df)} rows × 6)")

    summaries = []
    all_rows = []

    if args.include_base:
        print("\n" + "=" * 60)
        print("Scoring BASE (no adapter)")
        print("=" * 60)
        df = evaluate_model(base, flat_ds, tokenizer.pad_token_id,
                            args.batch_size, label="base", device=args.device)
        s = summarize("base", df)
        s_str = (f"  chosen={s['chosen_score']:+.4f}  "
                 f"rejected={s['rejected_score']:+.4f}  "
                 f"Δ={s['delta']:+.4f}  pref_acc={s['pref_acc']:.4f}")
        print(s_str)
        summaries.append(s)
        df["checkpoint"] = "base"
        all_rows.append(df)

    # Attach all adapters as named adapters; swap with set_adapter().
    adapter_names = []
    for i, path in enumerate(args.adapters):
        name = Path(path).name or f"ckpt_{i}"
        if name in adapter_names:
            name = f"{name}_{i}"
        adapter_names.append(name)

    print(f"\nAttaching {len(args.adapters)} adapters: {adapter_names}")
    model = PeftModel.from_pretrained(base, args.adapters[0],
                                      adapter_name=adapter_names[0])
    for path, name in zip(args.adapters[1:], adapter_names[1:]):
        model.load_adapter(path, adapter_name=name)
    model.eval()

    for path, name in zip(args.adapters, adapter_names):
        print("\n" + "=" * 60)
        print(f"Scoring {name}  ({path})")
        print("=" * 60)
        model.set_adapter(name)
        df = evaluate_model(model, flat_ds, tokenizer.pad_token_id,
                            args.batch_size, label=name, device=args.device)
        s = summarize(name, df, path=path)
        s_str = (f"  chosen={s['chosen_score']:+.4f}  "
                 f"rejected={s['rejected_score']:+.4f}  "
                 f"Δ={s['delta']:+.4f}  pref_acc={s['pref_acc']:.4f}")
        print(s_str)
        summaries.append(s)
        df["checkpoint"] = name
        all_rows.append(df)

    print("\n" + "=" * 86)
    print(f"SCORE TREND  (n={n}, valid={Path(args.valid_parquet).name})")
    print("=" * 86)
    print(f"{'checkpoint':<24}{'chosen':>12}{'rejected':>12}"
          f"{'Δ':>12}{'pref_acc':>12}")
    print("-" * 86)
    for s in summaries:
        print(f"{s['checkpoint']:<24}"
              f"{s['chosen_score']:>+12.4f}"
              f"{s['rejected_score']:>+12.4f}"
              f"{s['delta']:>+12.4f}"
              f"{s['pref_acc']:>12.4f}")

    print(
        "\nReading the trend:\n"
        "  chosen_score RISING  → SFT is pushing P(chosen) up (the goal)\n"
        "  rejected_score also rising → LM is generally lifting both → SFT\n"
        "                                 isn't discriminating, just shifting\n"
        "  pref_acc rising      → relative ordering improves\n"
        "  flat across ckpts    → training didn't move the model on this metric\n"
    )

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summaries).to_csv(out, index=False)
        print(f"saved trend summary to: {out}")

        rows_out = out.with_name(out.stem + "_rows.csv")
        pd.concat(all_rows, ignore_index=True).to_csv(rows_out, index=False)
        print(f"saved per-row breakdown to: {rows_out}")


if __name__ == "__main__":
    main()