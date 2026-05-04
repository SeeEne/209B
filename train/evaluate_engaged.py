"""
evaluate_engaged.py

Engagement-aware evaluation for the length=3 contrastive model.

Instead of fighting OneRec's held-out video_test (whose uids are disjoint
from the master table — no engagement labels available), we evaluate
directly on the held-out 10% of contrastive_dataset_v1/valid.parquet.

Each valid row already carries the same positive/negative split that
build_contrastive_dataset.py uses at training time:
  - chosen_sids   : 3 items the user gave a positive engagement signal
                    (longview / like / follow / forward)
  - rejected_sids : 3 items the user did NOT engage with positively
                    (no signal, or not_interested=1)

So we can compute Recall@K on BOTH:
  - recall_chosen   : how many engaged GT items appear in the model's top-K
  - recall_rejected : how many non-engaged items appear in the model's top-K
Difference = the model's preference accuracy at the *generation* level.

A model that genuinely learned the contrastive task should have
  recall_chosen > recall_rejected.
A baseline that doesn't discriminate should have them roughly equal.

Usage:
    python train/evaluate_engaged.py \
        --model_path runs/contrastive_v1_smoke/merged \
        --valid_parquet data/contrastive_dataset_v1/valid.parquet \
        --template model/qwen3_soft_switch.jinja2 \
        --n 1000 --num_beams 32 --topk 96 \
        --output_csv runs/eval_engaged_smoke_v1.csv
"""

import argparse
import re
import warnings
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

from dataset import SYSTEM_PROMPT, build_history_text, sid_to_core_text
from utils import resolve_template

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
    """Same prompt construction as ContrastiveDatasetLength3."""
    hist_text = build_history_text(hist_sids, max_hist=max_hist)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": hist_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ) + "<|sid_begin|>"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="Local checkpoint dir or HF repo id.")
    p.add_argument("--valid_parquet",
                   default="data/contrastive_dataset_v1/valid.parquet",
                   help="Held-out valid set with chosen_sids / rejected_sids "
                        "per row, built by build_contrastive_dataset.py.")
    p.add_argument("--template", default=None)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--num_beams", type=int, default=32)
    p.add_argument("--topk", type=int, default=None,
                   help="Defaults to --num_beams.")
    p.add_argument("--max_new_tokens", type=int,
                   default=MAX_NEW_TOKENS_LENGTH3)
    p.add_argument("--max_hist", type=int, default=512)
    p.add_argument("--cache_implementation",
                   choices=["dynamic", "offloaded"], default="dynamic")
    p.add_argument(
        "--subsample_seed", type=int, default=67890,
        help="Seed for subsampling valid to --n rows. Default 67890 matches "
             "the seed used by --max_eval_samples in train_contrastive*, "
             "so this script's evaluated subset == the training-time eval "
             "subset (same 1k samples).",
    )
    p.add_argument("--output_csv", default=None)
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

    print(f"Loading tokenizer + model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                              trust_remote_code=True)
    tokenizer.chat_template = template_path.read_text(encoding="utf-8")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
        attn_implementation="sdpa",
    )
    model.eval()
    for attr in ["temperature", "top_p", "top_k"]:
        if hasattr(model.generation_config, attr):
            try:
                setattr(model.generation_config, attr, None)
            except Exception:
                pass

    rows = []
    for i in tqdm(range(n), desc=f"Evaluating {n} (engagement-aware)"):
        row = valid_df.iloc[i]
        hist_sids = row["hist_sids"]
        chosen_sids = row["chosen_sids"]      # 3 items × [a, b, c]
        rejected_sids = row["rejected_sids"]

        prompt = build_eval_prompt(tokenizer, hist_sids, args.max_hist)

        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
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
        chosen_set = set(chosen_strs)
        rejected_set = set(rejected_strs)

        hits_c = len(chosen_set & topk_set)
        hits_r = len(rejected_set & topk_set)

        rows.append({
            "row_idx": i,
            "uid": int(row["uid"]) if "uid" in row.index else -1,
            "num_pred_unique": len(topk),
            "num_chosen": len(chosen_strs),
            "num_rejected": len(rejected_strs),
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

    res = pd.DataFrame(rows)

    rc = res["recall_chosen"].mean()
    rr = res["recall_rejected"].mean()
    pc = res["pass_chosen"].mean()
    pr = res["pass_rejected"].mean()

    print("\n" + "=" * 60)
    print(f"SUMMARY  (model={args.model_path})")
    print("=" * 60)
    print(f"N evaluated                  = {len(res)}")
    print(f"top-K (after pooling+dedup)  = {args.topk}")
    print()
    print(f"{'metric':<22}{'CHOSEN':>12}{'REJECTED':>12}{'Δ (C - R)':>14}")
    print(f"{'-' * 60}")
    print(f"{'recall@' + str(args.topk):<22}"
          f"{rc:>12.4f}{rr:>12.4f}{rc - rr:>+14.4f}")
    print(f"{'pass@' + str(args.topk):<22}"
          f"{pc:>12.4f}{pr:>12.4f}{pc - pr:>+14.4f}")
    print()
    print("Δ > 0  →  model ranks engaged items higher than non-engaged.")
    print("Δ ≈ 0  →  model doesn't discriminate (typical for untrained base).")
    print("Δ < 0  →  model anti-learned (worse than random for our task).")

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(out, index=False)
        print(f"\nsaved to: {out}")


if __name__ == "__main__":
    main()
