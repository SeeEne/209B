"""
compare_models.py

Diagnose whether contrastive training broke generation by directly comparing
base vs trained model outputs on the same benchmark prompts.

For each sample:
  1. Show top-K logits at the position right after `<|sid_begin|>` — i.e.
     what each model thinks the first SID token should be. If the trained
     model has crashed the SID-token probabilities, you'll see non-SID
     tokens dominating its top-K.
  2. Show beam-search generations (max_new_tokens=3) from both models.
     Mark each with ✓ if it produced a valid SID pattern, ✗ otherwise.

Both models are loaded onto the same GPU (~7 GB total for 2x bf16 1.7B).
"""

import argparse
import json
import re

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CORE_SID_PATTERN = re.compile(r"<s_a_\d+><s_b_\d+><s_c_\d+>")


def flatten_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            it.get("text", "") for it in content
            if isinstance(it, dict) and it.get("type") == "text"
        )
    return str(content)


def make_prompt(tok, messages):
    norm = [{"role": m.get("role", "user"),
             "content": flatten_content(m.get("content", ""))} for m in messages]
    return tok.apply_chat_template(
        norm, tokenize=False, add_generation_prompt=True
    ) + "<|sid_begin|>"


def load(path, template_path, name):
    print(f"[load] {name}: {path}")
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tok.chat_template = open(template_path).read()
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa",
    ).eval()
    return tok, model


def top_logits(model, tok, prompt, k):
    ids = tok(prompt, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        out = model(**ids)
    last = out.logits[0, -1]
    v, i = last.topk(k)
    return [(tok.decode([idx.item()]), val.item()) for val, idx in zip(v, i)]


def gen(model, tok, prompt, num_beams, max_new_tokens=3):
    ids = tok(prompt, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        out = model.generate(
            **ids, max_new_tokens=max_new_tokens, do_sample=False,
            num_beams=num_beams, num_return_sequences=num_beams,
            early_stopping=True, use_cache=True,
            pad_token_id=tok.pad_token_id,
        )
    plen = ids["input_ids"].shape[1]
    return [tok.decode(s[plen:], skip_special_tokens=False) for s in out]


def extract_all_sids(text):
    return CORE_SID_PATTERN.findall(text or "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="model/OneRec-1.7B")
    p.add_argument("--trained", default="runs/contrastive_v0_smoke/merged")
    p.add_argument("--template", default="model/qwen3_soft_switch.jinja2")
    p.add_argument("--benchmark",
                   default="data/OpenOneRec/benchmark_data/video/video_test.parquet")
    p.add_argument("--n_prompts", type=int, default=3)
    p.add_argument("--num_beams", type=int, default=8)
    p.add_argument("--top_k", type=int, default=15)
    p.add_argument("--max_new_tokens", type=int, default=13,
                   help="13 = 3 items × 3 SID tokens + 2 separators × 2 tokens. "
                        "Pass 3 to mimic length=1 evaluation (1 item per beam).")
    args = p.parse_args()

    df = pd.read_parquet(args.benchmark)
    tok_b, m_b = load(args.base,    args.template, "base   ")
    tok_t, m_t = load(args.trained, args.template, "trained")

    for i in range(args.n_prompts):
        row = df.iloc[i]
        msgs = (json.loads(row["messages"]) if isinstance(row["messages"], str)
                else row["messages"])
        meta = (json.loads(row["metadata"]) if isinstance(row["metadata"], str)
                else row["metadata"])
        prompt = make_prompt(tok_b, msgs)

        # Pull GT SIDs as a set for hit-marking inside generations.
        gt_sids = extract_all_sids(meta["answer"])
        gt_set = set(gt_sids)

        print(f"\n{'=' * 78}")
        print(f"PROMPT #{i}  uid={meta.get('uid')}")
        print(f"GT items ({len(gt_sids)} total): {gt_sids[:6]}"
              + ("..." if len(gt_sids) > 6 else ""))

        print(f"\n--- BASE: top-{args.top_k} logits after '<|sid_begin|>' ---")
        for s, l in top_logits(m_b, tok_b, prompt, args.top_k):
            mark = "★" if s.startswith("<s_a_") else " "
            print(f"  {mark} {l:8.3f}  {repr(s)}")

        print(f"\n--- TRAINED: top-{args.top_k} logits after '<|sid_begin|>' ---")
        for s, l in top_logits(m_t, tok_t, prompt, args.top_k):
            mark = "★" if s.startswith("<s_a_") else " "
            print(f"  {mark} {l:8.3f}  {repr(s)}")

        print(f"\n--- BASE: beam={args.num_beams}, "
              f"max_new_tokens={args.max_new_tokens} gens ---")
        base_pool = []
        for j, d in enumerate(gen(m_b, tok_b, prompt, args.num_beams,
                                  args.max_new_tokens)):
            sids = extract_all_sids(d)
            base_pool.extend(sids)
            hits = [s for s in sids if s in gt_set]
            mark = "✓" if sids else "✗"
            hitmark = f" 🎯×{len(hits)}" if hits else ""
            print(f"  {mark} {j}: {repr(d)}{hitmark}")

        print(f"\n--- TRAINED: beam={args.num_beams}, "
              f"max_new_tokens={args.max_new_tokens} gens ---")
        trained_pool = []
        for j, d in enumerate(gen(m_t, tok_t, prompt, args.num_beams,
                                  args.max_new_tokens)):
            sids = extract_all_sids(d)
            trained_pool.extend(sids)
            hits = [s for s in sids if s in gt_set]
            mark = "✓" if sids else "✗"
            hitmark = f" 🎯×{len(hits)}" if hits else ""
            print(f"  {mark} {j}: {repr(d)}{hitmark}")

        # Pool-level summary: dedup, count overlap with GT.
        base_pool_dedup = list(dict.fromkeys(base_pool))
        trained_pool_dedup = list(dict.fromkeys(trained_pool))
        base_hits = gt_set & set(base_pool_dedup)
        trained_hits = gt_set & set(trained_pool_dedup)
        print(f"\n--- POOL SUMMARY ---")
        print(f"  base    : {len(base_pool_dedup)} unique SIDs, "
              f"{len(base_hits)}/{len(gt_set)} GT hit")
        print(f"  trained : {len(trained_pool_dedup)} unique SIDs, "
              f"{len(trained_hits)}/{len(gt_set)} GT hit")
        if trained_pool_dedup:
            print(f"  trained top-10: {trained_pool_dedup[:10]}")


if __name__ == "__main__":
    main()
