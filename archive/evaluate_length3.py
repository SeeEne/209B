"""
evaluate_length3.py

Order-invariant evaluation for the length=3 contrastive model.

Each beam generates a 13-token answer = 3 items (3 SID tokens per item, 2
separator tokens between consecutive items). We extract all SIDs from every
beam, pool them across beams (preserving beam-rank order), deduplicate, and
take the top-K candidates. Match against the GT items as a SET — order is
ignored.

This differs from evaluate_origin.py in two ways:
    1. max_new_tokens = 13 instead of 3 (so each beam emits 3 items, not 1).
    2. Predictions are pooled across beams (beam 1's 3 items, then beam 2's,
       etc.) and deduplicated, instead of taking just the first SID per beam.

Usage:
    python train/evaluate_length3.py \
        --model_path runs/contrastive_v1_smoke/merged \
        --benchmark data/OpenOneRec/benchmark_data/video/video_test.parquet \
        --template model/qwen3_soft_switch.jinja2 \
        --n 100 \
        --output_csv runs/eval_smoke_length3.csv
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

from utils import resolve_template

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

CORE_SID_PATTERN = re.compile(r"<s_a_\d+><s_b_\d+><s_c_\d+>")
SID_BLOCK_PATTERN = re.compile(r"<\|sid_begin\|>.*?<\|sid_end\|>")

# How many new tokens a beam emits to cover 3 items:
#   3 SID tokens × 3 items + 2 separators × 2 tokens between = 13
MAX_NEW_TOKENS_LENGTH3 = 13


# ---------------------------------------------------------------------------
# Helpers (same as evaluate_origin.py)
# ---------------------------------------------------------------------------


def flatten_content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def normalize_messages(messages):
    return [
        {"role": m.get("role", "user"),
         "content": flatten_content_to_text(m.get("content", ""))}
        for m in messages
    ]


def extract_first_core_sid(text):
    m = CORE_SID_PATTERN.search(text or "")
    return m.group(0) if m else None


def extract_all_core_sids(text):
    """Pull every <s_a_*><s_b_*><s_c_*> in order from a string."""
    return CORE_SID_PATTERN.findall(text or "")


def dedup_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="OpenOneRec/OneRec-1.7B",
                        help="Local checkpoint dir or HF repo id.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--template", default=None)
    parser.add_argument("--n", type=int, default=100,
                        help="Number of samples (use -1 for full set)")
    parser.add_argument("--num_beams", type=int, default=32,
                        help="Beam width during generation.")
    parser.add_argument("--topk", type=int, default=None,
                        help="How many top candidate SIDs to keep after pooling "
                             "across beams. Defaults to --num_beams (matches "
                             "the K=32 reporting convention of length=1 eval).")
    parser.add_argument("--max_new_tokens", type=int,
                        default=MAX_NEW_TOKENS_LENGTH3,
                        help="13 covers 3 items × (3 SID tokens) + 2 × 2-token "
                             "<sid_end><sid_begin> separators.")
    parser.add_argument(
        "--cache_implementation",
        choices=["dynamic", "offloaded"],
        default="dynamic",
    )
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.topk is None:
        args.topk = args.num_beams

    warnings.filterwarnings("ignore")
    hf_logging.set_verbosity_error()

    benchmark_path = Path(args.benchmark)
    assert benchmark_path.exists(), f"benchmark not found: {benchmark_path}"
    template_path = resolve_template(args.template)

    bench_df = pd.read_parquet(benchmark_path)
    n = len(bench_df) if args.n < 0 else min(args.n, len(bench_df))

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
    for i in tqdm(range(n), desc=f"Evaluating {n} samples (length=3)"):
        row = bench_df.iloc[i]
        metadata = (json.loads(row["metadata"])
                    if isinstance(row["metadata"], str) else row["metadata"])
        messages = (json.loads(row["messages"])
                    if isinstance(row["messages"], str) else row["messages"])

        prompt = tokenizer.apply_chat_template(
            normalize_messages(messages),
            tokenize=False,
            add_generation_prompt=True,
        ) + "<|sid_begin|>"

        gt_blocks = SID_BLOCK_PATTERN.findall(metadata["answer"])
        gt_ids = [extract_first_core_sid(x) for x in gt_blocks]
        gt_ids = [x for x in gt_ids if x is not None]

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

        # Pool all SIDs from all beams in beam-rank order.
        # For each beam (sorted high-prob first), append its 3 SIDs
        # in the order they appear in the generation.
        flat_preds = []
        for beam_text in decoded:
            flat_preds.extend(extract_all_core_sids(beam_text))
        # Dedup preserving order, then keep top-K.
        topk = dedup_preserve_order(flat_preds)[:args.topk]

        # Order-invariant matching against the GT set.
        gt_set = set(gt_ids)
        topk_set = set(topk)
        hits = len(gt_set & topk_set)
        recall = hits / len(gt_ids) if gt_ids else 0.0
        passk = bool(gt_set & topk_set)
        position1_pass = (gt_ids[0] in topk_set) if gt_ids else False

        rows.append({
            "row_idx": i,
            "num_gt": len(gt_ids),
            "num_pred_unique": len(topk),
            "hits": hits,
            "recall": recall,
            "passk": passk,
            "position1_pass": position1_pass,
            "top5_pred": topk[:5],
            "top5_gt": gt_ids[:5],
            "uid": metadata.get("uid"),
            "uuid": metadata.get("uuid"),
        })

        del inputs, outputs, decoded
        if (i + 1) % 5 == 0 or i == 0:
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"  [mem@{i+1:>3d}] alloc={alloc:5.2f}G  "
                  f"reserved={reserved:5.2f}G  peak={peak:5.2f}G")
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()

    res = pd.DataFrame(rows)
    print("\n===== SUMMARY (length=3, order-invariant) =====")
    print(f"N             = {n}")
    print(f"num_beams     = {args.num_beams}")
    print(f"top-K (after pooling+dedup) = {args.topk}")
    print(f"recall@{args.topk}    = {res['recall'].mean():.4f}")
    print(f"pass@{args.topk}      = {res['passk'].mean():.4f}")
    print(f"position1_pass = {res['position1_pass'].mean():.4f}")

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(out, index=False)
        print(f"saved to: {out}")


if __name__ == "__main__":
    main()
