"""
diagnose/checkpoint_recall_trend.py

Run the engagement-aware eval (same logic as train/evaluate_engaged.py)
across multiple LoRA checkpoints to track whether recall_chosen is rising
or falling during training.

The point: with --sft_weight 0.1 we want to confirm that chosen recall
ACTUALLY GOES UP across checkpoints, not just that the DPO margin grows
(margin can grow by lowering both chosen and rejected, which is what pure
GRPO did and is the failure mode this run is supposed to fix).

Implementation: load base ONCE, attach all adapters as named PEFT adapters,
switch between them with set_adapter(). Memory ≈ base + 3×LoRA ≈ 3.5 GB.

Usage (from project root):
    python diagnose/checkpoint_recall_trend.py \
        --base model/OneRec-1.7B \
        --adapters runs/dpo_grpo_smoke/checkpoint-600 \
                   runs/dpo_grpo_smoke/checkpoint-800 \
                   runs/dpo_grpo_smoke/checkpoint-1000 \
        --include_base \
        --valid_parquet data/contrastive_dataset_v1/valid.parquet \
        --template model/qwen3_soft_switch.jinja2 \
        --n 1000 --num_beams 32 --topk 96 \
        --output_csv runs/checkpoint_trend_dpo_smoke.csv
"""

import argparse
import re
import sys
import warnings
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

# Reuse train/dataset.py + train/utils.py (same prompt construction as
# evaluate_engaged.py — keeps eval comparable).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "train"))
from dataset import SYSTEM_PROMPT, build_history_text, sid_to_core_text  # noqa: E402
from utils import resolve_template  # noqa: E402

CORE_SID_PATTERN = re.compile(r"<s_a_\d+><s_b_\d+><s_c_\d+>")
MAX_NEW_TOKENS_LENGTH3 = 13


def extract_all_core_sids(text):
    return CORE_SID_PATTERN.findall(text or "")


def dedup_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_eval_prompt(tokenizer, hist_sids, max_hist):
    hist_text = build_history_text(hist_sids, max_hist=max_hist)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": hist_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ) + "<|sid_begin|>"


def evaluate(model, tokenizer, valid_df, args, label):
    rows = []
    for i in tqdm(range(len(valid_df)), desc=f"eval {label}"):
        row = valid_df.iloc[i]
        hist_sids = row["hist_sids"]
        chosen_sids = row["chosen_sids"]
        rejected_sids = row["rejected_sids"]

        prompt = build_eval_prompt(tokenizer, hist_sids, args.max_hist)

        inputs = tokenizer(prompt, return_tensors="pt",
                           add_special_tokens=True)
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=args.num_beams,
                num_return_sequences=args.num_beams,
                early_stopping=True,
                use_cache=True,
                cache_implementation=args.cache_implementation,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(outputs[:, prompt_len:],
                                         skip_special_tokens=False)

        flat_preds = []
        for beam_text in decoded:
            flat_preds.extend(extract_all_core_sids(beam_text))
        topk = dedup_preserve_order(flat_preds)[:args.topk]
        topk_set = set(topk)

        chosen_strs = [sid_to_core_text(s) for s in chosen_sids]
        rejected_strs = [sid_to_core_text(s) for s in rejected_sids]

        hits_c = len(set(chosen_strs) & topk_set)
        hits_r = len(set(rejected_strs) & topk_set)

        rows.append({
            "row_idx": i,
            "uid": int(row["uid"]) if "uid" in row.index else -1,
            "hits_chosen": hits_c,
            "hits_rejected": hits_r,
            "recall_chosen": hits_c / max(1, len(chosen_strs)),
            "recall_rejected": hits_r / max(1, len(rejected_strs)),
            "pass_chosen": hits_c > 0,
            "pass_rejected": hits_r > 0,
        })

        del inputs, outputs, decoded
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def summarize(label, df, path=None):
    rc = df["recall_chosen"].mean()
    rr = df["recall_rejected"].mean()
    pc = df["pass_chosen"].mean()
    pr = df["pass_rejected"].mean()
    return {
        "checkpoint": label,
        "path": path or "",
        "n": len(df),
        "recall_chosen": rc,
        "recall_rejected": rr,
        "delta_recall": rc - rr,
        "pass_chosen": pc,
        "pass_rejected": pr,
        "delta_pass": pc - pr,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True,
                   help="Local path or HF id of the base OneRec model.")
    p.add_argument("--adapters", nargs="+", required=True,
                   help="Paths to LoRA adapter / Trainer checkpoint dirs. "
                        "Order matters — used as the trend axis.")
    p.add_argument("--include_base", action="store_true",
                   help="Also evaluate base (no adapter) as a reference row.")
    p.add_argument("--valid_parquet",
                   default="data/contrastive_dataset_v1/valid.parquet")
    p.add_argument("--template", default=None)
    p.add_argument("--n", type=int, default=1000,
                   help="Smaller default than evaluate_engaged.py — this is a "
                        "quick trend check, not a final number. Use --n 5000 "
                        "if you need decisive numbers.")
    p.add_argument("--num_beams", type=int, default=32)
    p.add_argument("--topk", type=int, default=None,
                   help="Defaults to --num_beams.")
    p.add_argument("--max_new_tokens", type=int,
                   default=MAX_NEW_TOKENS_LENGTH3)
    p.add_argument("--max_hist", type=int, default=512)
    p.add_argument("--cache_implementation",
                   choices=["dynamic", "offloaded"], default="dynamic")
    p.add_argument("--subsample_seed", type=int, default=67890,
                   help="Same default as evaluate_engaged.py so the "
                        "evaluated subset matches its --n 5000 run "
                        "(or a prefix thereof at smaller n).")
    p.add_argument("--output_csv", default=None,
                   help="Where to save the trend summary (one row per "
                        "checkpoint) + per-row breakdowns.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.topk is None:
        args.topk = args.num_beams

    warnings.filterwarnings("ignore")
    hf_logging.set_verbosity_error()

    template_path = resolve_template(args.template)

    print(f"Loading valid set from {args.valid_parquet} ...")
    valid_df = pd.read_parquet(args.valid_parquet).reset_index(drop=True)
    n_full = len(valid_df)
    n = n_full if args.n < 0 else min(args.n, n_full)
    print(f"  {n_full:,} total rows, evaluating {n}")
    if n < n_full:
        rng = torch.Generator().manual_seed(args.subsample_seed)
        indices = torch.randperm(n_full, generator=rng)[:n].tolist()
        valid_df = valid_df.iloc[indices].reset_index(drop=True)

    print(f"Loading tokenizer + base model from {args.base} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base,
                                              trust_remote_code=True)
    tokenizer.chat_template = template_path.read_text(encoding="utf-8")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
        attn_implementation="sdpa",
    )
    base.eval()
    for attr in ["temperature", "top_p", "top_k"]:
        if hasattr(base.generation_config, attr):
            try:
                setattr(base.generation_config, attr, None)
            except Exception:
                pass

    summaries = []
    all_rows = []  # row-level breakdowns, tagged with checkpoint label

    if args.include_base:
        print("\n" + "=" * 60)
        print("Evaluating BASE (no adapter)")
        print("=" * 60)
        df = evaluate(base, tokenizer, valid_df, args, label="base")
        s = summarize("base", df)
        summaries.append(s)
        df["checkpoint"] = "base"
        all_rows.append(df)
        print(f"  recall_chosen={s['recall_chosen']:.4f}  "
              f"rejected={s['recall_rejected']:.4f}  "
              f"Δrec={s['delta_recall']:+.4f}")

    # Attach all adapters as named adapters so we can swap with set_adapter.
    adapter_names = []
    for i, path in enumerate(args.adapters):
        # Use checkpoint dir name (e.g. "checkpoint-600") as adapter_name;
        # fall back to ckpt_{i} if name collision.
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
        print(f"Evaluating {name}  ({path})")
        print("=" * 60)
        model.set_adapter(name)
        df = evaluate(model, tokenizer, valid_df, args, label=name)
        s = summarize(name, df, path=path)
        summaries.append(s)
        df["checkpoint"] = name
        all_rows.append(df)
        print(f"  recall_chosen={s['recall_chosen']:.4f}  "
              f"rejected={s['recall_rejected']:.4f}  "
              f"Δrec={s['delta_recall']:+.4f}")

    # Trend table
    print("\n" + "=" * 82)
    print(f"CHECKPOINT TREND  (n={n}, top-K={args.topk}, "
          f"valid={Path(args.valid_parquet).name})")
    print("=" * 82)
    print(f"{'checkpoint':<22}{'rec_chosen':>12}{'rec_rejected':>14}"
          f"{'Δrec':>10}{'pass_C':>10}{'pass_R':>10}{'Δpass':>10}")
    print("-" * 82)
    for s in summaries:
        print(f"{s['checkpoint']:<22}{s['recall_chosen']:>12.4f}"
              f"{s['recall_rejected']:>14.4f}"
              f"{s['delta_recall']:>+10.4f}"
              f"{s['pass_chosen']:>10.4f}{s['pass_rejected']:>10.4f}"
              f"{s['delta_pass']:>+10.4f}")

    print(
        "\nGoal: recall_chosen RISING across checkpoints (SFT anchor working)\n"
        "      Δrec > 0 throughout (DPO discriminating chosen above rejected)\n"
        "      Reference: baseline (no FT) ≈ 0.0093 chosen / 0.0163 rejected "
        "at n=5000\n"
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
