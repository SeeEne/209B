"""
debug_recall.py

Run on official benchmark data (video_test.parquet) to verify our
generation + evaluation pipeline matches Table 4 numbers.

If this script reproduces ~0.05 Pass@1 and ~0.17 Pass@32, our code is correct.
If not, there's a bug to fix.

Usage:
    python debug_recall.py                # default 20 samples
    python debug_recall.py --n 100        # more samples
"""

import argparse
import json
import re
import random

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "OpenOneRec/OneRec-1.7B"
BENCH_PATH = "data/OpenOneRec/benchmark_data/video/video_test.parquet"
SID2PID_PATH = "data/OpenOneRec/benchmark_data/sid2pid.json"
DEVICE = "mps"

CODE_MUL_1 = 8192 * 8192
CODE_MUL_2 = 8192
NUM_BEAMS = 32
NUM_RETURN_SEQ = 32
MAX_NEW_TOKENS = 3

SID_PATTERN = re.compile(r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")

# ---------------------------------------------------------------------------
# SID→PID
# ---------------------------------------------------------------------------

def load_sid2pid(path):
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def sid_to_pid(c1, c2, c3, sid2pid_map):
    key = c1 * CODE_MUL_1 + c2 * CODE_MUL_2 + c3
    pid_list = sid2pid_map.get(key)
    if not pid_list:
        return 0
    max_count = max(info["count_after_downsample"] for info in pid_list)
    candidates = [info.get("pid", info.get("iid", 0))
                  for info in pid_list if info["count_after_downsample"] == max_count]
    return random.choice(candidates)

# ---------------------------------------------------------------------------
# Prompt building — use benchmark messages directly
# ---------------------------------------------------------------------------

def build_prompt_from_benchmark(row, tokenizer, device, sid_begin_id):
    """Use the EXACT messages from benchmark, no reconstruction."""
    msgs = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]
    sys_text = msgs[0]["content"][0]["text"] if isinstance(msgs[0]["content"], list) else msgs[0]["content"]
    usr_text = msgs[1]["content"][0]["text"] if isinstance(msgs[1]["content"], list) else msgs[1]["content"]

    messages = [
        {"role": "system", "content": sys_text},
        {"role": "user",   "content": usr_text},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(prompt_text, return_tensors="pt")
    prompt_ids = encoded["input_ids"]
    sid_token = torch.tensor([[sid_begin_id]], dtype=torch.long)
    prompt_ids = torch.cat([prompt_ids, sid_token], dim=1)
    return prompt_ids.to(device)

# ---------------------------------------------------------------------------
# Generation + parse
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_and_parse(prompt_ids, model, tokenizer, sid2pid_map):
    n_prompt = prompt_ids.shape[1]

    outputs = model.generate(
        prompt_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=NUM_BEAMS,
        num_return_sequences=NUM_RETURN_SEQ,
        do_sample=False,
    )

    seen = set()
    unique_pids = []

    for i in range(outputs.shape[0]):
        new_tokens = outputs[i, n_prompt:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        m = SID_PATTERN.search(text)
        if m is None:
            continue
        c1, c2, c3 = int(m.group(1)), int(m.group(2)), int(m.group(3))
        pid = sid_to_pid(c1, c2, c3, sid2pid_map)
        if pid == 0 or pid in seen:
            continue
        seen.add(pid)
        unique_pids.append(pid)

    return unique_pids

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="number of samples")
    args = parser.parse_args()

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, device_map=DEVICE, trust_remote_code=True
    )
    model.eval()

    sid_begin_id = tokenizer.convert_tokens_to_ids("<|sid_begin|>")
    print(f"<|sid_begin|> token ID: {sid_begin_id}")

    print("Loading sid2pid...")
    sid2pid_map = load_sid2pid(SID2PID_PATH)
    print(f"  {len(sid2pid_map):,} entries")

    print("Loading benchmark data...")
    bench = pd.read_parquet(BENCH_PATH)
    bench = bench.sample(n=min(args.n, len(bench)), random_state=42).reset_index(drop=True)
    print(f"  {len(bench)} samples")

    # --- Detailed debug on first sample ---
    print("\n" + "=" * 60)
    print("DEBUG: first sample detailed output")
    print("=" * 60)
    row0 = bench.iloc[0]
    meta0 = json.loads(row0["metadata"]) if isinstance(row0["metadata"], str) else row0["metadata"]
    gt_pids = meta0["answer_pid"]
    print(f"Ground truth PIDs ({len(gt_pids)}): {gt_pids}")

    prompt_ids = build_prompt_from_benchmark(row0, tokenizer, DEVICE, sid_begin_id)
    print(f"Prompt length: {prompt_ids.shape[1]} tokens")
    print(f"Last 5 tokens: {prompt_ids[0, -5:].tolist()}")
    for tid in prompt_ids[0, -5:].tolist():
        print(f"  {tid} -> {repr(tokenizer.decode([tid]))}")

    # Generate with small beam for debug
    print(f"\nGenerating (num_beams={NUM_BEAMS}, return={NUM_RETURN_SEQ})...")
    n_prompt = prompt_ids.shape[1]
    with torch.no_grad():
        outputs = model.generate(
            prompt_ids, max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS, num_return_sequences=NUM_RETURN_SEQ,
            do_sample=False,
        )
    print(f"Output shape: {outputs.shape}")
    print(f"\nAll {outputs.shape[0]} beam outputs:")
    for i in range(outputs.shape[0]):
        new_toks = outputs[i, n_prompt:]
        ids = new_toks.tolist()
        decoded = tokenizer.decode(new_toks, skip_special_tokens=False)
        m = SID_PATTERN.search(decoded)
        sid_str = f"({m.group(1)},{m.group(2)},{m.group(3)})" if m else "NO_MATCH"
        pid = 0
        if m:
            pid = sid_to_pid(int(m.group(1)), int(m.group(2)), int(m.group(3)), sid2pid_map)
        hit = "HIT!" if pid in gt_pids else ""
        print(f"  [{i:2d}] ids={ids}  decoded={repr(decoded):<45s}  sid={sid_str:<20s}  pid={pid:<12d}  {hit}")

    candidates = generate_and_parse(prompt_ids, model, tokenizer, sid2pid_map)
    gt_set = set(gt_pids)
    print(f"\nUnique candidates: {len(candidates)}")
    print(f"Pass@1:  {candidates[0] in gt_set if candidates else False}")
    print(f"Pass@32: {bool(set(candidates[:32]) & gt_set)}")

    # --- Run on all samples ---
    print("\n" + "=" * 60)
    print(f"Running on {len(bench)} benchmark samples...")
    print("=" * 60)

    pass1_hits = 0
    pass32_hits = 0
    recall32_sum = 0.0

    for idx in range(len(bench)):
        row = bench.iloc[idx]
        meta = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
        gt_set = set(meta["answer_pid"])

        prompt_ids = build_prompt_from_benchmark(row, tokenizer, DEVICE, sid_begin_id)
        candidates = generate_and_parse(prompt_ids, model, tokenizer, sid2pid_map)

        p1 = bool(candidates and candidates[0] in gt_set)
        p32 = bool(set(candidates[:32]) & gt_set)
        r32 = len(set(candidates[:32]) & gt_set) / len(gt_set) if gt_set else 0.0

        pass1_hits += p1
        pass32_hits += p32
        recall32_sum += r32

        if (idx + 1) % 5 == 0 or idx == 0:
            print(f"  [{idx+1:3d}/{len(bench)}] "
                  f"Pass@1={pass1_hits/(idx+1):.4f}  "
                  f"Pass@32={pass32_hits/(idx+1):.4f}  "
                  f"Recall@32={recall32_sum/(idx+1):.4f}  "
                  f"(this: cands={len(candidates)}, p1={p1}, p32={p32})")

    n = len(bench)
    print(f"\n{'='*60}")
    print(f"FINAL (n={n})")
    print(f"  Pass@1    : {pass1_hits/n:.4f}  (official: 0.0496)")
    print(f"  Pass@32   : {pass32_hits/n:.4f}  (official: 0.1710)")
    print(f"  Recall@32 : {recall32_sum/n:.4f}  (official: 0.0272)")


if __name__ == "__main__":
    main()
