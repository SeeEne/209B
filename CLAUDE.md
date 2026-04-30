# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project (current direction)

CS 209B course project based on the **OpenOneRec** open-source dataset/framework (Kuaishou, 2025). The project replaces OneRec's separately-trained Reward Model with **direct user behavior signals** as the supervision source for preference alignment, using **contrastive learning** (not DPO — see "Method pivot" below).

**Core research question:**
> In offline recommendation, can user behavior signals (longview / like / follow / forward) directly replace the Reward Model when constructing preference pairs for alignment training?

**Why this is novel.** OneRec's IPA module trains a separate Reward Model to score beam-search candidates because in their **online** setting user behaviors only arrive *after* recommendation. We work **offline** on logged data — every behavior signal is already in the parquet — so we can construct positive/negative pairs directly from behavior and skip the Reward Model entirely.

**Method pivot (DPO → contrastive).** The project originally targeted DPO. After EDA we abandoned DPO for two reasons documented in [notebook/eda_ms2.ipynb](notebook/eda_ms2.ipynb) and [oneRec/eda_findings.md](oneRec/eda_findings.md):
1. Explicit `not_interested` is too sparse (0.06% item rate; ~515 users with both pos+neg) for DPO.
2. Our top-m / bottom-m construction yields *synthetic* chosen sequences that never existed in the real log; DPO's likelihood-based loss penalizes this off-policy shift.

The current method is a **pairwise contrastive loss** on individual items (not sequences):
```
score(item | history) = (1/3) * Σ_t log P(item_token_t | history, item_<t)
L = -log_softmax([score+/τ, score-/τ])[0]
```
This sidesteps the synthetic-sequence problem (items, not sequences) and tolerates implicit negatives (no-signal items, which are abundant).

## Data

Dataset: `OpenOneRec/OpenOneRec-RecIF` (HF, gated). Already downloaded — minimal viable subset only:

```
data/OpenOneRec/
├── onerec_bench_release.parquet      # master training table, 162,074 rows, 25 cols
├── video_ad_pid2sid.parquet          # 15.9M video/ad pid → semantic ID (length-3 tuple)
├── product_pid2sid.parquet           # 2.1M product pid → semantic ID
└── benchmark_data/
    ├── video/video_test.parquet      # 38,781 rows  (the test set we evaluate on)
    ├── ad/ad_test.parquet            # 27,677 rows
    ├── product/product_test.parquet  # 27,910 rows
    ├── sid2pid.json
    └── sid2iid.json
```

Plus a derived training set built from the master table:

```
data/contrastive_dataset_v0/
├── train.parquet                      # 125,031 (history, chosen_sid, rejected_sid) triples
├── valid.parquet                      # 13,892 triples (10% holdout)
└── meta.json                          # build config + skip counts
```

**Important schema notes:**
- Repo SFT scripts only consume `split=0` rows — but in the HF release every row has `split=0`, so the filter is effectively a no-op.
- `hist_video_*` / `target_video_*` behavior columns are `list<int64>` aligned 1:1 to `*_pid` (per-item labels, not aggregate counts).
- `hist_ad_pid`, `hist_goods_pid`, `hist_longview_video_list` are `list<double>` (NaN-padded) — cast to int after filtering.
- `sid` is a `list<int64>` of length 3.

## Repo layout

```
project/
├── CLAUDE.md                              # this file
├── build_contrastive_dataset.py           # builds contrastive_dataset_v0 from the master table
├── notebook/
│   ├── EDA.md                             # MS2 research-design spec (historical)
│   ├── EDA_报告.md                         # MS2 findings summary (Chinese)
│   ├── eda_data_health.py                 # MS2 Script 1
│   ├── eda_behavior_signals.py            # MS2 Script 2
│   ├── eda_dpo_feasibility.py             # MS2 Script 3 (DPO feasibility — kept for record)
│   ├── eda_ms2.ipynb                      # MS2 narrative notebook (DPO → contrastive pivot story)
│   ├── eda_story.ipynb                    # earlier narrative draft
│   ├── ms3.ipynb                          # MS3 deliverable: EDA on contrastive set + baseline + pipeline
│   ├── ms3_baseline.ipynb                 # standalone baseline run on video_test.parquet (GPU)
│   └── outputs/                           # text reports from EDA scripts
├── train/
│   ├── README.md                          # how to train + evaluate
│   ├── dataset.py                         # ContrastiveDataset (builds OneRec-format prompts)
│   ├── train_contrastive.py               # LoRA contrastive FT via HF Trainer + PEFT
│   └── evaluate_origin.py                 # official-protocol Recall@K / Pass@K eval
├── data/                                  # not in git
│   ├── OpenOneRec/                        # downloaded HF dataset
│   └── contrastive_dataset_v0/            # built by build_contrastive_dataset.py
├── test_eda.ipynb                         # teammate's reference baseline (read-only)
└── oneRec/
    ├── *.pdf                              # OneRec tech report + dataset notes
    ├── training_plan.md                   # current training plan (LoRA + contrastive)
    └── eda_findings.md                    # MS2 EDA findings (history + DPO→contrastive pivot)
```

## Current status (2026-04-30)

**MS2 (EDA) — DONE.** Reports in [notebook/outputs/](notebook/outputs/), narrative in [notebook/eda_ms2.ipynb](notebook/eda_ms2.ipynb).
- Data is clean: 162,074 split=0 rows, 100% pid→sid coverage, 0 behavior/pid alignment mismatches, 156,245 usable users.
- Behavior signals semantically validated: like/follow/forward lift longview rate 1.36–1.88×; not_interested suppresses to 0.66×.
- Critical finding that triggered the DPO → contrastive pivot: target-side `not_interested` fires on only 0.06% of items.

**Contrastive dataset v0 — DONE.** Built by [build_contrastive_dataset.py](build_contrastive_dataset.py), `prediction_length=1`:
- Positive item: `longview=1` OR `like=1` OR `follow=1` OR `forward=1`
- Negative item: `not_interested=1` OR all 5 signals = 0
- 138,923 pairs total (125,031 train / 13,892 valid), 85.7% user coverage.

**Baseline evaluation — DONE.** OneRec-1.7B (no fine-tuning) on `video_test.parquet[:100]`:
- Recall@32 = 0.0231, Pass@32 = 0.13, Pass@1 = 0.06
- Reproduces official Table 4 numbers within sample-size variance — eval pipeline confirmed correct.
- Code: [notebook/ms3_baseline.ipynb](notebook/ms3_baseline.ipynb), packaged as CLI in [train/evaluate_origin.py](train/evaluate_origin.py).

**Training pipeline — READY (LoRA + HF Trainer + PEFT).** See [train/README.md](train/README.md).
- `train_contrastive.py` defaults `--model_path` to `OpenOneRec/OneRec-1.7B` (auto-pulled from HF Hub on first run).
- LoRA targets `q/k/v/o_proj` + `gate/up/down_proj` (Qwen-3 attention + MLP), `r=16`, `alpha=32`.
- Eval-time metric driving best-checkpoint selection: `pref_acc` on the valid set (fraction where chosen_score > rejected_score).
- Optionally `--merge_and_save` to write a fully-merged checkpoint that `evaluate_origin.py` can load directly.

**Next steps:**
1. Run LoRA contrastive FT on a CUDA GPU.
2. Re-run `evaluate_origin.py` on the trained checkpoint, compare Recall@32 / Pass@32 against baseline.
3. Optional ablations (see MS3 §6.3): signal granularity, `prediction_length` ∈ {1,3,5}, temperature τ.
4. Future-work metric design: preference-aware Recall/Pass that weights hits by engagement signal (MS3 §5.4).

## Dependencies

Python 3.10+. Key packages:
- Data / EDA: `pandas`, `pyarrow`, `numpy`, `huggingface_hub`, `matplotlib`
- Training: `torch`, `transformers`, `peft`, `accelerate`, `tqdm`

## Commands

EDA scripts (each standalone, run from project root):
```bash
python notebook/eda_data_health.py
python notebook/eda_behavior_signals.py
python notebook/eda_dpo_feasibility.py
```

Build contrastive dataset:
```bash
python build_contrastive_dataset.py --length 1
# → data/contrastive_dataset_v0/{train,valid}.parquet + meta.json
```

Train (single CUDA GPU; both model and template auto-resolved):
```bash
python train/train_contrastive.py \
    --train_parquet data/contrastive_dataset_v0/train.parquet \
    --valid_parquet data/contrastive_dataset_v0/valid.parquet \
    --output_dir runs/contrastive_v0_lora \
    --merge_and_save
```

Evaluate (baseline or trained checkpoint):
```bash
# baseline
python train/evaluate_origin.py \
    --benchmark data/OpenOneRec/benchmark_data/video/video_test.parquet \
    --n 100 --output_csv runs/eval_baseline.csv

# trained
python train/evaluate_origin.py \
    --model_path runs/contrastive_v0_lora/merged \
    --benchmark data/OpenOneRec/benchmark_data/video/video_test.parquet \
    --n 100 --output_csv runs/eval_contrastive_v0.csv
```

**Auto-resolution.** `--model_path` defaults to `OpenOneRec/OneRec-1.7B` and `from_pretrained` pulls from the HF Hub on first run. `--template` is resolved by [train/utils.py](train/utils.py) `resolve_template()` in this order: explicit CLI flag → `<project_root>/oneRec/qwen3_soft_switch.jinja2` → `~/.cache/onerec_template/qwen3_soft_switch.jinja2` → download from `https://raw.githubusercontent.com/Kuaishou-OneRec/OpenOneRec/main/benchmarks/benchmark/tasks/v1_0/qwen3_soft_switch.jinja2`. Override the URL with `ONEREC_TEMPLATE_URL` if upstream layout changes.

HF auth: token lives at `~/.cache/huggingface/token`. The OpenOneRec-RecIF *dataset* is gated — request access on the HF repo page first. The OneRec-1.7B *model* is ungated. `huggingface-cli` is not always on PATH; use `python -c "from huggingface_hub import login; login()"` if re-auth needed.

## Conventions

- All EDA filters to `split=0` only. Never touch `benchmark_data/` for training analysis — it is the held-out test set, completely disjoint from the master table.
- EDA scripts emit text reports to `notebook/outputs/`; don't dump giant intermediate parquets.
- Three-script EDA structure (data_health → behavior_signals → dpo_feasibility) is a deliberate narrative — keep them independent so each can be re-run in isolation. The "dpo_feasibility" script is kept under its original name for git history; its findings now motivate the contrastive method.
- Eval is **always** done with `train/evaluate_origin.py` on `video_test.parquet`, matching the official OneRec protocol (beam search 32, max_new_tokens=3, SID-string-level matching). Do not invent a custom split — use the published benchmark so numbers are comparable.
- SID matching is at the **string level** (`<s_a_X><s_b_Y><s_c_Z>`), not at the PID level. The SID→PID mapping is many-to-one, so PID-level matching introduces ambiguity.
