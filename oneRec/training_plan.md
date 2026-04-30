# Training Plan — Contrastive Fine-Tuning Pipeline

> **Note (2026-04-30).** This plan was originally written for DPO. After EDA we
> pivoted to contrastive learning (see [eda_findings.md](eda_findings.md) and
> [../notebook/eda_ms2.ipynb](../notebook/eda_ms2.ipynb)). The current plan
> below reflects the contrastive method.

## Base model

Use **`OpenOneRec/OneRec-1.7B`** (HF, 4.29 GB, ungated). It has gone through Pretraining → SFT → Distillation → GRPO-RL, so it is a competent generative recommender out of the box. We add a **LoRA** adapter on top for behavior-signal preference alignment — full fine-tuning is unnecessary for a 105k-pair contrastive run on a 1.7B base.

`evaluate_origin.py` and `train_contrastive.py` default `--model_path` to this HF id, so the model is auto-pulled from the Hub on first run if not already cached locally.

**Do not use:**
- `OneRec-1.7B-pro` — Pro variants include Kuaishou internal data; distribution doesn't match our open-source dataset, makes the baseline noisy.
- `OneRec-1.7B-pretrain` — pretrain-only, would require us to redo SFT (large extra workload, no benefit).
- 8B variants — needs multi-GPU, overkill for an academic ablation.

Other repos in the OpenOneRec HF org: `OneRec-tokenizer`, `OneRec-{8B,8B-pro,8B-pro-pretrain,8B-pretrain,1.7B-pro-pretrain}`.

## Full pipeline

```
1. EDA Scripts 1/2/3                                          [DONE]
2. build_contrastive_dataset.py
   → data/contrastive_dataset_v0/{train,valid}.parquet (138,923 pairs)
   → meta.json                                                [DONE]
3. Baseline eval of OneRec-1.7B on video_test.parquet
   → Recall@32 = 0.023, Pass@32 = 0.13, Pass@1 = 0.06
   → matches official Table 4                                 [DONE]
4. LoRA contrastive fine-tuning (single CUDA GPU)
   - HF Trainer + PEFT, custom compute_loss with pairwise InfoNCE
   - LoRA on q/k/v/o_proj + gate/up/down_proj, r=16
   - bf16 + gradient checkpointing
   - eval-time metric: pref_acc on valid set (chosen_score > rejected_score)
   - --merge_and_save writes full merged weights for downstream eval
   - ~125k pairs × 2 epochs ≈ 4-8 h on a single A100 40 GB
5. Eval trained checkpoint on benchmark_data/video/video_test.parquet
   - same script (evaluate_origin.py), point --model_path to <output_dir>/merged
   - compare Recall@32 / Pass@32 / Pass@1 vs baseline
6. (optional) Ablations: signal granularity, prediction_length ∈ {1,3,5}, τ
7. (future work) Preference-aware metrics that weight hits by engagement signal
```

## Loss

For each (history, chosen_item, rejected_item):

```
score(item | history) = (1/3) * Σ_t log P(item_token_t | history, item_<t)
                                  ─────────────────────────────────
                                  averaged over the 3 SID tokens

L = -log_softmax([score+/τ, score-/τ])[0]
  = log(1 + exp(-(score+ - score-)/τ))
```

This is pairwise InfoNCE with one negative — equivalent to the Bradley-Terry / BPR ranking loss with a temperature scaling. We score only the 3 SID tokens (`<s_a_X><s_b_Y><s_c_Z>`); `<|sid_end|>` is omitted because it is deterministic and would dilute the learning signal.

The `chosen` and `rejected` sequences share the same prompt prefix, so we pack them as a `(2B, L)` batch and compute both scores in one forward pass.

## Compute footprint

- Disk: 4.3 GB base model + ~700 MB dataset + LoRA adapters (~50 MB each) + optional merged checkpoints (~3.4 GB each).
- VRAM: LoRA on 1.7B base in bf16 with gradient checkpointing fits comfortably on a single A100 40 GB at `per_device_batch_size=2, grad_accum=8` (effective batch 16).
- Wall time: 4–8 h per training run, depending on `max_hist`, `max_total_len`, and grad accumulation.

## Why LoRA + HF Trainer (not full FT, not custom loop)

- LoRA: ~10–20M trainable params (vs 1.7B), faster, less memory, well-documented behavior on preference-alignment tasks. Standard choice for DPO/RLHF-style work.
- HF Trainer: handles bf16, gradient accumulation, scheduler, periodic checkpointing, eval, and best-model selection (`metric_for_best_model="eval_pref_acc"`) out of the box. Custom `compute_loss` and `prediction_step` plug in cleanly.

## Rationale for not pretraining or re-SFT

OneRec used 96M interactions × hundreds of GPUs × weeks; that's impossible at academic scale. Re-SFT on a published post-SFT checkpoint adds large engineering effort with no scientific return. Behavior-signal contrastive alignment on the post-SFT checkpoint is the maximum-leverage, minimum-risk path to the experiment we actually care about.

Always evaluate against the vanilla `OneRec-1.7B` baseline using the same `benchmark_data/video/video_test.parquet`, not a custom split — keeps numbers comparable to the OneRec tech report (Table 4) and to anyone else who runs `evaluate_origin.py`.
