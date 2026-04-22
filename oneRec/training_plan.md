# Training Plan — Base Model and DPO Pipeline

## Base model

Use **`OpenOneRec/OneRec-1.7B`** (HF, 4.29 GB, ungated, post-SFT Standard version). This is the right choice because:

- It has **already gone through SFT** — DPO can start immediately, no need to redo SFT
- Standard variant trained on the same open-source data we have, distribution-aligned
- 1.7B fits on a single A100 40GB / A6000 48GB even with the reference model copy DPO needs
- Built on Qwen3-1.7B with Itemic-Text Alignment + Co-Pretraining + Multi-task SFT

**Do not use:**
- `OneRec-1.7B-pro` — Pro variants include Kuaishou internal data; distribution doesn't match our open-source dataset, makes baseline noisy
- `OneRec-1.7B-pretrain` — pretrain-only, would require us to do SFT ourselves (large extra workload, no benefit)
- 8B variants — needs multi-GPU, overkill for an academic ablation

Other repos in the OpenOneRec HF org: `OneRec-tokenizer` (itemic tokenizer alone), `OneRec-{8B,8B-pro,8B-pro-pretrain,8B-pretrain,1.7B-pro-pretrain}`.

## Full pipeline

```
1. EDA Scripts 1/2/3                                          [DONE]
2. build_dpo_dataset.py
   → data/dpo_dataset/{train,valid}.parquet (~105K pairs)
   → meta.json with build config + stats
3. huggingface-cli download OpenOneRec/OneRec-1.7B (~4.3 GB)
4. DPO training
   - trl DPOTrainer (or repo's RL script if it exists)
   - single A100 40GB / A6000 48GB
   - ~105K pair × 3 epoch ≈ 4-12 h depending on seq length / batch
5. Evaluation on benchmark_data/video/video_test.parquet (38,781 samples)
   - metrics: Recall@10, Pass@32, Pass@1
   - baseline: vanilla OneRec-1.7B inference (no DPO)
6. (optional) Add Arm 3 vs Arm 2 negative ablation as a paper sub-section
```

## Compute footprint

- Disk: 4.3 GB model + ~700 MB dataset (parquet snappy) + checkpoints
- VRAM: DPO needs policy + reference model in memory, 1.7B × 2 ≈ 8 GB params + activations + optimizer → 40 GB GPU is comfortable at batch 4-8
- Wall time: 4-12 h per training run

## Rationale

This plan deliberately avoids self-pretraining (impossible at academic scale: OneRec used 96M interactions × hundreds of GPUs × weeks) and self-SFT (large engineering effort with no scientific return). DPO on a published post-SFT checkpoint is the maximum-leverage, minimum-risk path to the experiment we actually care about.

Target trl `DPOTrainer` first; only fall back to OpenOneRec's repo scripts if there's a model-format mismatch. Always evaluate against the vanilla `OneRec-1.7B` baseline using the same `benchmark_data/video/` test set, not a custom split.