# Training & Evaluation

## Files

- `dataset.py` — `ContrastiveDataset` for `contrastive_dataset_v0`. Builds OneRec chat-format prompts (system + history) and tokenizes chosen / rejected SIDs sharing the same prefix.
- `train_contrastive.py` — LoRA contrastive fine-tuning of OneRec-1.7B via HuggingFace Trainer + PEFT.
- `evaluate_origin.py` — official-protocol Recall@K / Pass@K eval on `video_test.parquet` (mirrors `test_eda.ipynb`). Use the same script before and after training.
- `utils.py` — `resolve_template()`: locates or downloads the `qwen3_soft_switch.jinja2` chat template (CLI flag → `<project>/oneRec/` → `~/.cache/onerec_template/` → GitHub raw fetch).
- `debug_pipeline.py` — CPU-only smoke test of the whole stack (deps → template → tokenizer → dataset → collation → model load → LoRA → forward → loss → backward). Stops before training. Run this on your laptop before launching a GPU job.

## Smoke test (CPU, no training)

```bash
# Full check — loads OneRec-1.7B in fp32 on CPU (~7 GB RAM, 1-2 min)
python train/debug_pipeline.py

# Data-only check — skip model load / forward / backward
python train/debug_pipeline.py --skip_model
```

Each step prints `✓ PASS` or `✗ FAIL`. The final summary lists everything; exit code is 0 only if all steps pass. Use `--skip_model` if your machine can't afford ~7 GB RAM for the fp32 base model — the data and tokenization checks alone catch most bugs before you ship a GPU job.

## Requirements

```
torch
transformers
peft
accelerate
pandas
pyarrow
tqdm
```

## Loss

Per pair `(history, chosen, rejected)`:

```
score(item | history) = (1/3) * Σ_t log P(item_token_t | history, item_<t)

L = -log_softmax([score+/τ, score-/τ])[0]
  = log(1 + exp(-(score+ - score-)/τ))
```

Reported during training: per-step loss + `pref_acc` (fraction where `score+ > score-`).

## Train (LoRA + HF Trainer)

Both `--model_path` and `--template` auto-resolve. The model defaults to `OpenOneRec/OneRec-1.7B` (auto-pulled from the HF Hub on first run). The template is found in this order:
1. `--template` if explicitly provided
2. `<project_root>/oneRec/qwen3_soft_switch.jinja2`
3. `~/.cache/onerec_template/qwen3_soft_switch.jinja2`
4. Downloaded from the OpenOneRec GitHub repo into (3) on first run

So the minimal invocation is:

```bash
python train/train_contrastive.py \
    --train_parquet data/contrastive_dataset_v0/train.parquet \
    --valid_parquet data/contrastive_dataset_v0/valid.parquet \
    --output_dir runs/contrastive_v0_lora \
    --merge_and_save
```

Full hyperparameter form:

```bash
python train/train_contrastive.py \
    --train_parquet data/contrastive_dataset_v0/train.parquet \
    --valid_parquet data/contrastive_dataset_v0/valid.parquet \
    --output_dir runs/contrastive_v0_lora \
    --epochs 2 \
    --per_device_batch_size 2 \
    --grad_accum 8 \
    --lr 2e-4 \
    --temperature 0.1 \
    --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
    --max_hist 512 --max_total_len 3072 \
    --merge_and_save
```

If the auto-fetch URL ever breaks, set `ONEREC_TEMPLATE_URL` or drop the file at `<project_root>/oneRec/qwen3_soft_switch.jinja2` manually.

Effective batch = `per_device_batch_size * grad_accum`. Gradient checkpointing on; bf16 weights/activations. LoRA targets `q/k/v/o_proj` and `gate/up/down_proj` (Qwen-3 attention + MLP).

`runs/contrastive_v0_lora/` will hold:
- `checkpoint-<step>/` — Trainer auto-checkpoints (rotated by `save_total_limit`)
- `adapter/` — LoRA weights only (~50MB)
- `merged/` — base + adapter merged into a full model (only with `--merge_and_save`)
- `config.json` — CLI args used

**Best-checkpoint selection**: `load_best_model_at_end=True` with `metric_for_best_model="eval_pref_acc"`, so the final model is whichever checkpoint had the highest preference accuracy on `valid.parquet`.

## Evaluate (before vs after)

```bash
# Baseline (auto-pulls model + template if not cached)
python train/evaluate_origin.py \
    --benchmark data/OpenOneRec/benchmark_data/video/video_test.parquet \
    --n 100 \
    --output_csv runs/eval_baseline.csv

# After contrastive training (use the merged checkpoint)
python train/evaluate_origin.py \
    --model_path runs/contrastive_v0_lora/merged \
    --benchmark data/OpenOneRec/benchmark_data/video/video_test.parquet \
    --n 100 \
    --output_csv runs/eval_contrastive_v0.csv
```

Use `--n -1` for the full 38,781-sample test set. If you skipped `--merge_and_save`, load the adapter manually:

```python
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(BASE_PATH, trust_remote_code=True, ...)
model = PeftModel.from_pretrained(base, "runs/contrastive_v0_lora/adapter")
model = model.merge_and_unload()
```

## Notes

- `max_hist=512` keeps the full user history (OneRec's release is already capped at 512). This matches the eval-time prompt distribution exactly. Lower it only if you're OOM — but mind that training and eval will then see different context lengths.
- `max_total_len=3072`: 512 history items × 5 tokens + chat template + 3 SID tokens ≈ 2600, with ~470 token headroom.
- The 3 SID tokens scored are `<s_a_X><s_b_Y><s_c_Z>` — `<|sid_end|>` is omitted (deterministic).
- Each example expands to 2 sequences (chosen + rejected) inside the model forward, so a `--per_device_batch_size 2` micro-batch processes 4 sequences at once.
- LoRA LR (~2e-4) is intentionally higher than full-FT LR (~1e-5) — LoRA adapters need a stronger update to move from zero init.
