# Training & Evaluation

Active code path: **DPO + SFT + group-normalized contrastive loss**.

For history of how we got here (length=1 contrastive → KL → length=3 → group
normalization → DPO+SFT), see [`../archive/EXPERIMENTS.md`](../archive/EXPERIMENTS.md).

## Files

- `dataset.py` — `SYSTEM_PROMPT`, `build_history_text`, `sid_to_core_text`, and the legacy `ContrastiveDataset` class. The current trainer defines its own `ContrastiveDatasetGRPO` inline; this file is kept because the helpers are shared.
- `utils.py` — `resolve_template()` for the OneRec chat template (CLI flag → project / cache → upstream GitHub).
- `train_contrastive_dpo_g_normalize.py` — **MAIN trainer**. Loss = L_dpo_grpo + sft_weight × L_sft.
- `evaluate_engaged.py` — **MAIN evaluator**. Reports recall_chosen, recall_rejected, and Δ on the held-out `contrastive_dataset_v1/valid.parquet`.
- `evaluate_origin.py` — Official OneRec Recall@K / Pass@K on `video_test.parquet`. Use as a comparison-to-paper reference; the headline number is from `evaluate_engaged.py`.

Older / experimental files (length=1, length=3 contrastive, GRPO-only, length=3 evaluator, CPU pipeline test) live in [`../archive/`](../archive/).

## Quick start: smoke (recommended first run)

```bash
bash run_dpo_smoke.sh   # from project root, ~5h on RTX 6000 Pro
```

Trains 5000 groups (= 15,000 pairs at G=3), then runs `evaluate_engaged.py` for the engagement-aware Δrecall / Δpass numbers. Logs to `runs/dpo_grpo_smoke.log`.

## Loss

Per pair `(history, chosen, rejected)`:

```
margin_p   = β × [(chosen_θ − chosen_ref) − (rejected_θ − rejected_ref)]
L_dpo_p    = softplus(−margin_p)                                     # per pair
```

GRPO-inspired group normalization (each user contributes G=3 pairs sharing a `group_id`):

```
L_grouped  = L_dpo_pair.view(num_groups, G)
std_g      = L_grouped.std(dim=-1).detach() + ε                       # ε=1e-3
L_dpo_grpo = (L_grouped / std_g).mean()
```

SFT anchor pushes chosen probability up in absolute terms — without it, the contrastive part can be minimized by *lowering* both chosen and rejected (rejected lower) instead of raising chosen:

```
L_sft   = −chosen_θ.mean()
L_total = L_dpo_grpo + sft_weight × L_sft
```

**No explicit KL by default.** DPO's reference baseline already saturates the sigmoid when trained drifts from ref, providing implicit KL. Set `--kl_weight > 0` only if you want extra constraint.

## Train

```bash
python train/train_contrastive_dpo_g_normalize.py \
    --model_path model/OneRec-1.7B \
    --template model/qwen3_soft_switch.jinja2 \
    --train_parquet data/contrastive_dataset_v1_grpo/train.parquet \
    --valid_parquet data/contrastive_dataset_v1_grpo/valid.parquet \
    --output_dir runs/<run_name> \
    --G 3 \
    --max_train_groups 50000 --max_eval_groups 2000 \
    --eval_steps 2500 --save_steps 2500 --logging_steps 50 \
    --per_device_batch_size 12 --grad_accum 1 \
    --lr 5e-5 --dpo_beta 0.1 --sft_weight 0.1 --kl_weight 0 \
    --merge_and_save
```

For smoke runs override:
- `--max_train_groups 5000` (15k pairs)
- `--eval_steps 200 --save_steps 200`
- `--logging_steps 25`

**Reference model is always loaded** (DPO requires ref scores). Adds ~3.4 GB VRAM.

`runs/<name>/` contents after training:
- `checkpoint-<step>/` — Trainer auto-checkpoints (rotated by `save_total_limit`)
- `adapter/` — best LoRA weights (`load_best_model_at_end=True`, `metric_for_best_model="eval_pref_acc"`)
- `merged/` — base + adapter merged into a full model (only with `--merge_and_save`)
- `config.json` — CLI args used

If you skipped `--merge_and_save`, run `merge_local.py` afterwards:

```bash
python merge_local.py \
    --base model/OneRec-1.7B \
    --adapter runs/<run_name>/adapter \
    --out runs/<run_name>/merged
```

## Evaluate

### Engagement-aware (primary metric)

`evaluate_engaged.py` runs on `contrastive_dataset_v1/valid.parquet` (held-out 10%, never seen during training). Each row has 3 chosen + 3 rejected; we compute recall against each set.

```bash
python train/evaluate_engaged.py \
    --model_path runs/<run_name>/merged \
    --valid_parquet data/contrastive_dataset_v1/valid.parquet \
    --template model/qwen3_soft_switch.jinja2 \
    --n 5000 --num_beams 32 --topk 96 \
    --output_csv runs/eval_engaged_<run_name>.csv
```

Output:
```
metric         CHOSEN    REJECTED     Δ (C - R)
recall@96      ...       ...          ...
pass@96        ...       ...          ...
```

**Δ > 0** means model ranks chosen items higher than rejected. **Baseline** (no FT) gives **Δ = −0.020** (biased toward rejected — OneRec was trained for next-shown). Goal: positive Δ AND `recall_chosen ≥ baseline 0.0093` (which is why we added the SFT anchor).

### OneRec Recall@K (secondary metric, comparable to paper)

```bash
python train/evaluate_origin.py \
    --model_path runs/<run_name>/merged \
    --benchmark data/OpenOneRec/benchmark_data/video/video_test.parquet \
    --template model/qwen3_soft_switch.jinja2 \
    --n 100 --output_csv runs/eval_origin_<run_name>.csv
```

Baseline (no FT): Recall@32 ≈ 0.025, Pass@32 ≈ 0.15. After DPO+SFT training, this number is **expected to be lower** than baseline — that's the whole point of switching to engagement-aware evaluation. Report both numbers for context.

## Notes

- `max_hist=512` keeps the full user history (matches eval-time prompt distribution).
- `max_total_len=3072`: 512 history items × 5 tokens + chat template + 3 SID tokens ≈ 2600, with ~470 token headroom.
- The 3 SID tokens scored are `<s_a_X><s_b_Y><s_c_Z>` — `<|sid_end|>` is omitted (deterministic).
- Each pair forwards 2 trained sequences (chosen + rejected) AND 2 ref sequences. So `--per_device_batch_size 12` (= 4 users × G=3) processes 24 trained + 24 ref sequences per micro-batch.
- DPO LR (5e-5) is intentionally lower than the contrastive era (2e-4) — DPO's ref baseline already provides a strong gradient direction; lower LR avoids overshoot.
- LoRA targets `q/k/v/o_proj` and `gate/up/down_proj` (Qwen-3 attention + MLP). r=16, α=32.
- `eps=1e-3` in the GRPO normalization caps the std-amplification factor at 1000× and avoids initial-step explosion when `std_g ≈ 0`.

## Hyperparameter cheat sheet

| Knob | Default | Tune up if … | Tune down if … |
|---|---|---|---|
| `--dpo_beta` | 0.1 | training too soft, model not learning | gradient explodes (β too large saturates sigmoid quickly) |
| `--sft_weight` | 0.1 | chosen recall **drops** vs baseline | chosen recall too dominant, Δ shrinking |
| `--kl_weight` | 0 | trained drifts too far from ref (rare with DPO) | (default 0 already off) |
| `--lr` | 5e-5 | loss flat after warmup | grad_norm consistently > 30 (clipped to 1.0) |
| `--G` | 3 | (G > 3 didn't help in our ablation; pairs in a group are too correlated) | (G < 3 loses normalization benefit) |
| `--per_device_batch_size` | 12 | VRAM < 50 GB used → can go to 24 (8 users × 3) | OOM (drop to 6 = 2 users × 3) |
