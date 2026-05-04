# Diagnostic tools

One-off scripts for inspecting models and verifying the pipeline,
separate from the production training/evaluation path.

| File | Purpose | When to use |
|---|---|---|
| `compare_models.py` | Side-by-side dump of (a) top-K logits at the first SID position and (b) beam-search generations, for two models on the same prompts. Marks each generation with ✓/✗ for SID validity and 🎯×N for GT hits. | After training, to diagnose whether a trained model has *mode-collapsed* (same SIDs across all users), *anti-aligned* (predicting opposite of GT), or just *shifted* (healthy distribution, different ranking). |
| `debug_recall.py` | Reproduces the OneRec paper's Table 4 baseline numbers (Pass@1 ≈ 0.05, Pass@32 ≈ 0.17) on a small sample of `video_test.parquet`. | Quick sanity check that our generation + matching code is correct. Run after upgrading transformers or changing the eval loop. Functionally a slimmer ancestor of `train/evaluate_origin.py`. |

## Quick usage

```bash
# compare base vs trained DPO model on 3 benchmark prompts
python diagnose/compare_models.py \
    --base model/OneRec-1.7B \
    --trained runs/dpo_grpo_smoke/merged \
    --template model/qwen3_soft_switch.jinja2 \
    --num_beams 8 \
    --max_new_tokens 13

# verify pipeline reproduces baseline
python diagnose/debug_recall.py --n 100
```

## Note on imports

These scripts were written when files lived at the project root and in
`train/`. After the May reorganization (see `archive/EXPERIMENTS.md`), some
imports may need a `sys.path` tweak or relative path update. They run from
the project root.
