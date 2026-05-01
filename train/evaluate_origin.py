"""
evaluate_origin.py

Standalone evaluation script for OneRec-style models on the official benchmark.
Reproduces the logic in test_eda.ipynb / ms3_baseline.ipynb.

Usage:
    python train/evaluate_origin.py \
        --model_path /path/to/checkpoint \
        --benchmark /path/to/video_test.parquet \
        --template /path/to/qwen3_soft_switch.jinja2 \
        --n 100 \
        --output_csv runs/eval_result.csv

After contrastive training, swap --model_path to the trained checkpoint and
compare against the baseline numbers.
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
# Regex patterns
# ---------------------------------------------------------------------------

SID_BLOCK_PATTERN = re.compile(r"<\|sid_begin\|>.*?<\|sid_end\|>")
CORE_SID_PATTERN = re.compile(r"<s_a_\d+><s_b_\d+><s_c_\d+>")

# ---------------------------------------------------------------------------
# Helpers
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="OpenOneRec/OneRec-1.7B",
                        help="HF repo id or local checkpoint dir. "
                             "Default pulls OneRec-1.7B from the Hub on first run; "
                             "after training, point this to runs/<exp>/merged")
    parser.add_argument("--benchmark", required=True,
                        help="Path to video_test.parquet (or ad_test, product_test)")
    parser.add_argument("--template", default=None,
                        help="Optional path to qwen3_soft_switch.jinja2. "
                             "If omitted, looks under <project>/oneRec/, "
                             "then ~/.cache/onerec_template/, then downloads "
                             "from the OpenOneRec GitHub repo.")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of samples (use -1 for full set)")
    parser.add_argument("--num_beams", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=3)
    parser.add_argument(
        "--cache_implementation",
        choices=["dynamic", "offloaded"],
        default="dynamic",
        help="dynamic = standard GPU KV cache (fast, may OOM at high "
             "num_beams). offloaded = KV cache on CPU, transferred per "
             "layer; ~1s/sample slower but uses ~5x less VRAM.",
    )
    parser.add_argument("--output_csv", default=None,
                        help="Where to write per-sample results CSV")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    hf_logging.set_verbosity_error()

    benchmark_path = Path(args.benchmark)
    assert benchmark_path.exists(), f"benchmark not found: {benchmark_path}"
    template_path = resolve_template(args.template)

    bench_df = pd.read_parquet(benchmark_path)
    n = len(bench_df) if args.n < 0 else min(args.n, len(bench_df))

    print(f"Loading tokenizer + model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.chat_template = template_path.read_text(encoding="utf-8")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load directly onto GPU to avoid the CPU->GPU double-allocation,
    # and force SDPA so beam search uses memory-efficient attention.
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
    for i in tqdm(range(n), desc=f"Evaluating {n} samples"):
        row = bench_df.iloc[i]
        metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
        messages = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]

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
        decoded = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=False)

        pred_ids = [extract_first_core_sid(t) for t in decoded]
        pred_ids = [x for x in pred_ids if x is not None]
        topk = pred_ids[:args.num_beams]

        hits = sum(1 for gt in gt_ids if gt in topk)
        recall = hits / len(gt_ids) if gt_ids else 0.0
        passk = any(gt in topk for gt in gt_ids) if gt_ids else False
        position1_pass = (gt_ids[0] in topk) if gt_ids else False

        rows.append({
            "row_idx": i,
            "num_gt": len(gt_ids),
            "num_pred": len(pred_ids),
            "hits": hits,
            "recall": recall,
            "passk": passk,
            "position1_pass": position1_pass,
            "top5_pred": pred_ids[:5],
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
    print("\n===== SUMMARY =====")
    print(f"N             = {n}")
    print(f"num_beams     = {args.num_beams}")
    print(f"recall@{args.num_beams}    = {res['recall'].mean():.4f}")
    print(f"pass@{args.num_beams}      = {res['passk'].mean():.4f}")
    print(f"position1_pass = {res['position1_pass'].mean():.4f}")

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(out, index=False)
        print(f"saved to: {out}")


if __name__ == "__main__":
    main()
