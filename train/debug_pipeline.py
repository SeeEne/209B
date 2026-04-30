"""
debug_pipeline.py

Smoke test the contrastive training pipeline end-to-end on CPU.

Stops before actual training but exercises everything else:
    Python / deps -> template resolution -> data files -> tokenizer -> dataset
    -> collation -> model load -> LoRA setup -> forward -> loss -> backward
    -> grad sanity check on LoRA params.

Each step prints a [STEP] header and either ✓ PASS or ✗ FAIL with a stack
trace, then a summary at the end.

Defaults are tiny (max_hist=8, max_total_len=256, batch=2, 1 forward) so the
1.7B model can run forward+backward on CPU in a couple of minutes. Pass
--skip_model if you don't have ~7 GB free RAM for the fp32 base model.

Usage:
    # full pipeline (loads OneRec-1.7B from HF on first run, ~7 GB RAM)
    python train/debug_pipeline.py

    # data + dataset + collation only, no model
    python train/debug_pipeline.py --skip_model

    # custom data
    python train/debug_pipeline.py --train_parquet path/to/train.parquet
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAIN_DIR))


# ---------------------------------------------------------------------------
# Step logger
# ---------------------------------------------------------------------------

class StepLogger:
    def __init__(self):
        self.passed = []
        self.failed = []

    def step(self, name, fn):
        print(f"\n[STEP] {name}")
        try:
            result = fn()
            print(f"  ✓ PASS")
            self.passed.append(name)
            return result
        except Exception as e:
            print(f"  ✗ FAIL: {type(e).__name__}: {e}")
            traceback.print_exc(limit=4)
            self.failed.append((name, repr(e)))
            return None


def section(title: str):
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_python():
    v = sys.version_info
    print(f"  python: {sys.version.split()[0]}  ({platform.platform()})")
    assert v >= (3, 10), f"need python >= 3.10, got {sys.version}"


def check_imports():
    required = [
        "torch", "transformers", "peft", "accelerate",
        "pandas", "pyarrow", "tqdm", "numpy",
    ]
    for name in required:
        try:
            mod = importlib.import_module(name)
        except ImportError as e:
            raise ImportError(f"missing dependency: {name}") from e
        version = getattr(mod, "__version__", "?")
        print(f"  {name:14s} {version}")


def check_device():
    import torch
    print(f"  torch:        {torch.__version__}")
    print(f"  CUDA avail:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    device:     {torch.cuda.get_device_name(0)}")
    if hasattr(torch.backends, "mps"):
        print(f"  MPS avail:    {torch.backends.mps.is_available()}")
    print(f"  num threads:  {torch.get_num_threads()}")


def find_train_parquet(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        assert p.exists(), f"not found: {p}"
        return p
    candidates = [
        PROJECT_ROOT / "data" / "contrastive_dataset_v0" / "train.parquet",
        PROJECT_ROOT / "data" / "contrastive_dataset_v0" / "valid.parquet",
    ]
    for c in candidates:
        if c.exists():
            print(f"  found: {c}")
            return c
    raise FileNotFoundError(
        "no contrastive dataset found. Run build_contrastive_dataset.py first, "
        "or pass --train_parquet."
    )


def load_tokenizer(model_path: str, template_path: Path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.chat_template = template_path.read_text(encoding="utf-8")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  vocab size: {tokenizer.vocab_size:,}")
    print(f"  pad token : {tokenizer.pad_token!r}  (id={tokenizer.pad_token_id})")
    print(f"  eos token : {tokenizer.eos_token!r}  (id={tokenizer.eos_token_id})")
    return tokenizer


def verify_special_tokens(tokenizer):
    """Confirm SID-related tokens exist and have stable single-token IDs."""
    must_exist = {
        "<|sid_begin|>": None,
        "<|sid_end|>": None,
        "<s_a_0>": None,
        "<s_a_8191>": None,
        "<s_b_0>": None,
        "<s_c_0>": None,
    }
    for tok in must_exist:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        assert len(ids) == 1, f"{tok!r} tokenizes to {len(ids)} tokens, expected 1: {ids}"
        must_exist[tok] = ids[0]
        print(f"  {tok:18s} -> id={ids[0]}")
    # Sanity: the codebook indices for s_a should be contiguous (8192 entries).
    ids_a = tokenizer.convert_tokens_to_ids([f"<s_a_{i}>" for i in [0, 1, 4096, 8190, 8191]])
    print(f"  s_a sample   ids: {ids_a}")
    assert all(i is not None and i >= 0 for i in ids_a), f"some s_a tokens unknown: {ids_a}"


def make_dataset(parquet, tokenizer, max_hist, max_total_len):
    from dataset import ContrastiveDataset
    ds = ContrastiveDataset(parquet, tokenizer, max_hist=max_hist,
                            max_total_len=max_total_len)
    print(f"  dataset size: {len(ds):,} pairs")
    return ds


def inspect_samples(dataset, tokenizer, n: int):
    print(f"  inspecting first {n} samples ...")
    for i in range(n):
        ex = dataset[i]
        L_c = ex["chosen_input_ids"].size(0)
        L_r = ex["rejected_input_ids"].size(0)
        plen = ex["prompt_len"]
        print(f"\n  --- sample {i} ---")
        print(f"    chosen_len   = {L_c}   (prompt + 3 SID tokens)")
        print(f"    rejected_len = {L_r}")
        print(f"    prompt_len   = {plen}")
        # Decode the 3 scored tokens for visual confirmation.
        chosen_sid = tokenizer.decode(
            ex["chosen_input_ids"][plen:plen + 3], skip_special_tokens=False)
        rejected_sid = tokenizer.decode(
            ex["rejected_input_ids"][plen:plen + 3], skip_special_tokens=False)
        prompt_tail = tokenizer.decode(
            ex["chosen_input_ids"][max(0, plen - 6):plen], skip_special_tokens=False)
        print(f"    prompt tail  : ...{prompt_tail!r}")
        print(f"    chosen   SID : {chosen_sid!r}")
        print(f"    rejected SID : {rejected_sid!r}")
        # Structural assertions
        assert L_c == L_r, f"chosen/rejected lengths differ ({L_c} vs {L_r})"
        # All three scored tokens should differ in chosen vs rejected — pure
        # signal for the contrastive loss.
        c_tok = ex["chosen_input_ids"][plen:plen + 3].tolist()
        r_tok = ex["rejected_input_ids"][plen:plen + 3].tolist()
        assert c_tok != r_tok, "chosen and rejected SID tokens are identical"


def deep_inspect_construction(parquet_path, dataset, tokenizer, max_hist):
    """
    Verify our training prompt is structurally identical to the official
    video_test.parquet prompt (same system text, same per-item SID wrapping,
    same trailing assistant + <|sid_begin|>).
    """
    import json
    import pandas as pd
    from dataset import (SYSTEM_PROMPT, build_history_text, sid_to_core_text,
                         sid_to_text)

    print("\n  --- raw row from contrastive_dataset_v0 (sample 0) ---")
    raw = pd.read_parquet(parquet_path).iloc[0]
    hist_sids = raw["hist_sids"]
    chosen_sids = raw["chosen_sids"]
    rejected_sids = raw["rejected_sids"]
    print(f"    uid           : {int(raw['uid'])}")
    print(f"    n_hist_items  : {len(hist_sids)}  (will be truncated to {max_hist})")
    print(f"    chosen_sids[0]: {list(chosen_sids[0])}  -> {sid_to_core_text(chosen_sids[0])}")
    print(f"    rejected_sids : {list(rejected_sids[0])}  -> {sid_to_core_text(rejected_sids[0])}")

    print("\n  --- expected wrapping for the LAST 3 history items ---")
    for s in hist_sids[-3:]:
        print(f"    {list(s)}  ->  {sid_to_text(s)}")

    print("\n  --- our reconstructed full prompt (last 200 chars) ---")
    hist_text = build_history_text(hist_sids, max_hist=max_hist)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": hist_text},
    ]
    our_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ) + "<|sid_begin|>"
    print(f"    [head]  {our_prompt[:120]!r}")
    print(f"    [tail]  ...{our_prompt[-200:]!r}")

    # Token-level structural checks on the dataset's tokenized output.
    ex = dataset[0]
    plen = ex["prompt_len"]
    chosen_ids = ex["chosen_input_ids"].tolist()
    rejected_ids = ex["rejected_input_ids"].tolist()

    print("\n  --- tokenized form (sample 0) ---")
    print(f"    prompt_len = {plen}, total_len = {len(chosen_ids)}")
    print(f"    last 6 prompt tokens (positions {plen-6}..{plen-1}):")
    for pos in range(max(0, plen - 6), plen):
        tid = chosen_ids[pos]
        print(f"      [{pos}] id={tid:6d}  -> {tokenizer.decode([tid])!r}")
    print(f"    chosen 3 SID tokens (positions {plen}..{plen+2}):")
    for k, pos in enumerate(range(plen, plen + 3)):
        tid = chosen_ids[pos]
        print(f"      [{pos}] id={tid:6d}  -> {tokenizer.decode([tid])!r}  "
              f"(expected level-{['a','b','c'][k]} = {chosen_sids[0][k]})")

    # Critical assertions.
    sid_begin_id = tokenizer.convert_tokens_to_ids("<|sid_begin|>")
    sid_end_id   = tokenizer.convert_tokens_to_ids("<|sid_end|>")
    assert chosen_ids[plen - 1] == sid_begin_id, (
        f"token before SID slot is {chosen_ids[plen-1]} "
        f"({tokenizer.decode([chosen_ids[plen-1]])!r}), expected <|sid_begin|>")

    n_begin = sum(1 for t in chosen_ids[:plen] if t == sid_begin_id)
    n_end   = sum(1 for t in chosen_ids[:plen] if t == sid_end_id)
    n_hist_in_prompt = min(len(hist_sids), max_hist)
    expected_begin = n_hist_in_prompt + 1   # n history + the trailing prompt token
    expected_end   = n_hist_in_prompt
    print(f"\n    <|sid_begin|> count in prompt: {n_begin}  (expected {expected_begin})")
    print(f"    <|sid_end|>   count in prompt: {n_end}    (expected {expected_end})")
    assert n_begin == expected_begin, "history items not wrapped with <|sid_begin|> correctly"
    assert n_end == expected_end, "history items not wrapped with <|sid_end|> correctly"

    # Chosen vs rejected: prompts identical, only the 3 trailing SID tokens differ.
    assert chosen_ids[:plen] == rejected_ids[:plen], "chosen/rejected prompts differ"
    diff_positions = [i for i in range(plen, plen + 3)
                      if chosen_ids[i] != rejected_ids[i]]
    assert len(diff_positions) == 3, (
        f"chosen and rejected differ in {len(diff_positions)}/3 SID positions; "
        f"expected all 3 to differ")
    print("    chosen vs rejected: identical prompts, all 3 SID tokens differ ✓")

    # Cross-check with the official benchmark's prompt format.
    test_path = (PROJECT_ROOT / "data" / "OpenOneRec" / "benchmark_data"
                 / "video" / "video_test.parquet")
    if test_path.exists():
        print("\n  --- cross-check vs official video_test.parquet sample 0 ---")
        test_row = pd.read_parquet(test_path).iloc[0]
        msgs = test_row["messages"]
        msgs = json.loads(msgs) if isinstance(msgs, str) else msgs

        def _flatten(c):
            if isinstance(c, str):
                return c
            return "".join(x.get("text", "") if isinstance(x, dict) else str(x)
                           for x in c)

        sys_txt = _flatten(msgs[0]["content"])
        usr_txt = _flatten(msgs[1]["content"])
        print(f"    official system : {sys_txt[:60]!r}")
        print(f"    our    system   : {SYSTEM_PROMPT[:60]!r}")
        if sys_txt.strip() == SYSTEM_PROMPT.strip():
            print("    system prompts MATCH ✓")
        else:
            print("    !! system prompts DIFFER — review dataset.py SYSTEM_PROMPT")

        # Tokenize the official user content the same way and compare wrapping.
        official_first_50 = usr_txt[:50]
        our_first_50 = hist_text[:50]
        print(f"    official user[:50]: {official_first_50!r}")
        print(f"    our    user[:50]: {our_first_50!r}  (different items, same wrapping pattern)")
        # Pattern check: must start with <|sid_begin|><s_a_
        assert usr_txt.startswith("<|sid_begin|><s_a_"), \
            "official user content has a different SID wrapping pattern!"
        assert hist_text.startswith("<|sid_begin|><s_a_"), \
            "our user content has a different SID wrapping pattern!"
        print("    user-content wrapping pattern MATCHES ✓")

        # Reconstruct what evaluate_origin.py does, tokenize, compare last tokens.
        norm_msgs = [{"role": m["role"], "content": _flatten(m["content"])} for m in msgs]
        official_prompt = tokenizer.apply_chat_template(
            norm_msgs, tokenize=False, add_generation_prompt=True
        ) + "<|sid_begin|>"
        official_ids = tokenizer(official_prompt, add_special_tokens=True)["input_ids"]
        print(f"    official prompt last 4 tokens: "
              f"{[tokenizer.decode([t]) for t in official_ids[-4:]]}")
        print(f"    our      prompt last 4 tokens: "
              f"{[tokenizer.decode([t]) for t in chosen_ids[plen-4:plen]]}")
        assert official_ids[-1] == sid_begin_id, \
            "official prompt does not end with <|sid_begin|>"
        assert chosen_ids[plen - 1] == sid_begin_id, \
            "our prompt does not end with <|sid_begin|>"
        print("    both prompts end with <|sid_begin|> at the same relative position ✓")
    else:
        print(f"\n  (skipping eval cross-check; {test_path} not present)")


def make_batch(dataset, batch_size, pad_token_id):
    from dataset import contrastive_collate
    items = [dataset[i] for i in range(batch_size)]
    batch = contrastive_collate(items, pad_token_id=pad_token_id)
    print(f"  input_ids      : {tuple(batch['input_ids'].shape)}    "
          f"(2B = {batch['input_ids'].size(0)})")
    print(f"  attention_mask : {tuple(batch['attention_mask'].shape)}")
    print(f"  prompt_lens    : {batch['prompt_lens'].tolist()}")
    return batch


def verify_batch_shape(batch):
    iid = batch["input_ids"]
    am = batch["attention_mask"]
    plens = batch["prompt_lens"]
    assert iid.shape == am.shape, "input_ids and attention_mask shape mismatch"
    bsz = iid.size(0)
    assert bsz % 2 == 0, "batch size must be 2B (chosen+rejected)"
    half = bsz // 2
    assert plens.size(0) == half, f"prompt_lens has {plens.size(0)} entries, expected {half}"
    # Each prompt_len should be < total length.
    assert (plens + 3 <= iid.size(1)).all(), "prompt_len + 3 SID tokens exceeds seq len"


def load_model_cpu(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM
    print(f"  loading {model_path} on CPU in fp32 (this may take 30-60s and ~7GB RAM) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  total params: {n_params:,}")
    return model


def apply_lora(model, target_modules):
    from peft import LoraConfig, TaskType, get_peft_model
    R = 8                # tiny adapter for the smoke test
    ALPHA = 16
    DROPOUT = 0.0
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=R,
        lora_alpha=ALPHA,
        lora_dropout=DROPOUT,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, cfg)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print("  ┌─ LoRA config ─────────────────────────────────")
    print(f"  │  rank (r)        : {R}")
    print(f"  │  alpha           : {ALPHA}     (scaling = alpha/r = {ALPHA/R:.2f})")
    print(f"  │  dropout         : {DROPOUT}")
    print(f"  │  target modules  : {target_modules}")
    print("  ├─ Parameter counts ────────────────────────────")
    print(f"  │  trainable       : {n_trainable:>13,}")
    print(f"  │  total           : {n_total:>13,}")
    print(f"  │  trainable %     : {100 * n_trainable / n_total:>13.4f} %")
    print("  └───────────────────────────────────────────────")
    return model


def run_forward(model, batch):
    import torch
    print("  running forward pass on CPU ...")
    with torch.no_grad():
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
    print(f"  logits shape: {tuple(out.logits.shape)}    (2B, L, V)")
    return out.logits


def run_loss(logits, batch):
    import torch
    from train_contrastive import compute_sid_logprobs, contrastive_loss
    chosen_s, rejected_s = compute_sid_logprobs(
        logits, batch["input_ids"], batch["prompt_lens"],
    )
    print(f"  chosen scores  : {chosen_s.tolist()}")
    print(f"  rejected scores: {rejected_s.tolist()}")
    margin = (chosen_s - rejected_s).mean().item()
    print(f"  mean margin (chosen - rejected): {margin:+.4f}")
    loss = contrastive_loss(chosen_s, rejected_s, temperature=0.1)
    print(f"  contrastive loss (τ=0.1): {loss.item():.4f}")
    assert torch.isfinite(loss), "loss is not finite"
    assert loss.item() >= 0, "loss should be non-negative"
    return chosen_s, rejected_s


def run_backward(model, batch):
    import torch
    from train_contrastive import compute_sid_logprobs, contrastive_loss
    print("  forward + backward (gradients flow into LoRA only) ...")
    model.train()
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    chosen_s, rejected_s = compute_sid_logprobs(
        out.logits, batch["input_ids"], batch["prompt_lens"],
    )
    loss = contrastive_loss(chosen_s, rejected_s, temperature=0.1)
    loss.backward()

    # All trainable params (LoRA) should have non-None .grad with a finite norm.
    n_tensors = 0
    n_with_grad = 0
    n_scalars = 0
    grad_norms = []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n_tensors += 1
        n_scalars += p.numel()
        if p.grad is not None:
            n_with_grad += 1
            grad_norms.append(p.grad.norm().item())
    print(f"  trainable tensors : {n_tensors}  (with grad: {n_with_grad})")
    print(f"  trainable scalars : {n_scalars:,}")
    assert n_with_grad == n_tensors, "some trainable tensors received no gradient"
    print(f"  grad norm range   : [{min(grad_norms):.2e}, {max(grad_norms):.2e}]")
    print(f"  grad norm mean    : {sum(grad_norms)/len(grad_norms):.2e}")
    print("  (note: debug uses r=8, targets={q_proj,v_proj} for speed. "
          "Production train_contrastive.py uses r=16 + full attn+MLP targets, "
          "≈ 18M trainable scalars on 1.7B base.)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="OpenOneRec/OneRec-1.7B")
    parser.add_argument("--template", default=None)
    parser.add_argument("--train_parquet", default=None,
                        help="If omitted, auto-finds data/contrastive_dataset_v0/train.parquet")
    parser.add_argument("--max_hist", type=int, default=8,
                        help="History items per example (small for CPU speed)")
    parser.add_argument("--max_total_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=2,
                        help="How many dataset samples to inspect verbatim")
    parser.add_argument("--skip_model", action="store_true",
                        help="Skip model load + forward/backward (data-only smoke test)")
    parser.add_argument("--lora_target_modules", nargs="+",
                        default=["q_proj", "v_proj"],
                        help="Trim LoRA targets in debug to keep memory low")
    args = parser.parse_args()

    log = StepLogger()

    section("1. Environment")
    log.step("Python version >= 3.10", check_python)
    log.step("Required packages installed", check_imports)
    log.step("Device info (CUDA / MPS / threads)", check_device)

    section("2. Resources")
    from utils import resolve_template
    template_path = log.step("Resolve qwen3_soft_switch.jinja2",
                             lambda: resolve_template(args.template))
    train_parquet = log.step("Locate train.parquet",
                             lambda: find_train_parquet(args.train_parquet))

    if template_path is None or train_parquet is None:
        section("Aborted: required resource missing")
        sys.exit(1)

    section("3. Tokenizer")
    tokenizer = log.step("Load tokenizer + chat template",
                        lambda: load_tokenizer(args.model_path, template_path))
    if tokenizer is None:
        sys.exit(1)
    log.step("Verify SID special tokens",
             lambda: verify_special_tokens(tokenizer))

    section("4. Dataset")
    dataset = log.step(
        "Build ContrastiveDataset",
        lambda: make_dataset(train_parquet, tokenizer,
                             args.max_hist, args.max_total_len),
    )
    if dataset is None:
        sys.exit(1)
    log.step("Inspect samples (lengths + decoded SIDs)",
             lambda: inspect_samples(dataset, tokenizer, args.num_samples))
    log.step("Deep verify prompt construction vs eval format",
             lambda: deep_inspect_construction(
                 train_parquet, dataset, tokenizer, args.max_hist))

    section("5. Collation")
    batch = log.step(
        "Pack chosen + rejected into (2B, L) batch",
        lambda: make_batch(dataset, args.batch_size, tokenizer.pad_token_id),
    )
    if batch is None:
        sys.exit(1)
    log.step("Verify batch shapes / prompt_len bounds",
             lambda: verify_batch_shape(batch))

    if args.skip_model:
        section("Model / forward / backward — SKIPPED (--skip_model)")
    else:
        section("6. Model")
        model = log.step("Load OneRec base model on CPU (fp32)",
                         lambda: load_model_cpu(args.model_path))
        if model is None:
            section("Aborted: model load failed")
            _summary(log)
            sys.exit(1)

        model = log.step("Wrap with LoRA adapters",
                         lambda: apply_lora(model, args.lora_target_modules))
        if model is None:
            _summary(log); sys.exit(1)

        section("7. Forward / Loss / Backward")
        logits = log.step("Forward pass (no grad)", lambda: run_forward(model, batch))
        if logits is None:
            _summary(log); sys.exit(1)
        log.step("Compute scores + contrastive loss",
                 lambda: run_loss(logits, batch))
        log.step("Backward pass + LoRA grad sanity",
                 lambda: run_backward(model, batch))

    _summary(log)
    sys.exit(0 if not log.failed else 1)


def _summary(log: StepLogger):
    section("Summary")
    print(f"PASSED: {len(log.passed)}")
    for name in log.passed:
        print(f"  ✓ {name}")
    if log.failed:
        print(f"\nFAILED: {len(log.failed)}")
        for name, err in log.failed:
            print(f"  ✗ {name}  :: {err}")
    else:
        print("\nAll steps passed. Pipeline is ready for GPU training.")


if __name__ == "__main__":
    main()
