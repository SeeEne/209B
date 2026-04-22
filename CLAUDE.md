# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project (current direction)

CS 1090a course project based on the **OpenOneRec** open-source dataset/framework (Kuaishou, 2025), focusing on **DPO data construction strategy**.

**Core research question** (from [notebook/EDA.md](notebook/EDA.md)):
> In offline recommendation, can user behavior signals directly replace the Reward Model when constructing DPO preference pairs? How do different signal granularities affect alignment?

**Why this is novel.** OpenOneRec's IPA module trains a separate Reward Model to score beam-search candidate sessions, because in their **online** setting user behaviors (like / follow / longview) only arrive *after* recommendation. We work **offline** on logged data — every behavior signal is already in the parquet — so we can compute reward directly from behavior and skip the Reward Model entirely. The contribution is a systematic ablation of signal granularity for the rule-based reward.

**Original 4-arm ablation was dropped** after EDA — see "Current status" below. The project now uses a **single reward formula** (Arm 2: `1.0·longview + 1.5·like + 2.0·follow + 1.5·forward`) with a gap threshold. See [notebook/EDA.md](notebook/EDA.md) §"实验设计调整" for the rationale.

Read [notebook/EDA.md](notebook/EDA.md) before touching anything — it is the source of truth for current research design, signal definitions, and reward formulas.

## Data

Dataset: `OpenOneRec/OpenOneRec-RecIF` (HF, gated). Already downloaded — minimal viable subset only:

```
data/OpenOneRec/
├── onerec_bench_release.parquet      # master training table, 162,074 rows, 25 cols
├── video_ad_pid2sid.parquet          # 15.9M video/ad pid → semantic ID (length-3 tuple)
├── product_pid2sid.parquet           # 2.1M product pid → semantic ID
└── benchmark_data/
    ├── video/video_test.parquet      # 38,781 rows
    ├── ad/ad_test.parquet            # 27,677 rows
    ├── product/product_test.parquet  # 27,910 rows
    ├── sid2pid.json
    └── sid2iid.json
```

Not downloaded (skip unless task expands): `pid2caption.parquet`, and the 5 other benchmark task directories (`interactive`, `label_cond`, `label_pred`, `item_understand`, `rec_reason`).

**Important schema notes** (from inspecting the parquet):
- Repo SFT scripts only consume `split=0` rows. All analyses must filter to `split=0` to avoid leaking benchmark users.
- `hist_video_*` behavior columns are `list<int64>` aligned to `hist_video_pid` (per-item labels, not aggregate counts). Same for `target_video_*`.
- `hist_ad_pid`, `hist_goods_pid`, `hist_longview_video_list` are `list<double>` (not int) because of NaN padding — cast to int after filtering NaNs.
- Mapping tables: `sid` is a `list<int64>` of length 3 (verify in EDA).

## Current status (2026-04-07)

EDA Scripts 1/2/3 completed. Reports in [notebook/outputs/](notebook/outputs/). Key findings drove a research-design adjustment — see [notebook/EDA.md](notebook/EDA.md) §"EDA 跑出来后的关键发现" and §"实验设计调整" for the full record. Highlights:

- **Data is clean**: 162,074 split=0 rows = full training pool, 100% pid→sid coverage, behavior/pid length alignment 0 mismatches, 156,245 usable users.
- **Behavior signals validated**: explicit-positive signals (like/follow/forward) lift longview rate 1.36-1.88×; not_interested suppresses it to 0.66× — semantic-consistency hypothesis fully supported.
- **Critical finding**: target-side `not_interested` is extremely sparse (0.06% item rate; 0.34% user coverage). This kills the original Arm 3 design.
- **Arm 2 vs Arm 3 chosen-subset agreement = 99.83%** — Arm 3 ablation effect ≤ ~500 samples.
- **Decision**: drop the 4-arm ablation. Use a single Arm 2 reward (`1.0·longview + 1.5·like + 2.0·follow + 1.5·forward`) with `gap > 1.5` filter → **105,383 DPO pairs** (Zephyr-DPO scale). Treat the Arm 3 vs Arm 2 finding as a negative result for the discussion section.
- **Pair construction**: per user, take the 10-item `target_video_pid`, score per item under Arm 2, top-5 = chosen, bot-5 = rejected, both reordered by original target time. One pair per user.
- **Base model**: download [`OpenOneRec/OneRec-1.7B`](https://huggingface.co/OpenOneRec/OneRec-1.7B) (4.29 GB, ungated, post-SFT Standard version). Skip Pro variants (internal-data drift) and pretrain-only variants (would need to redo SFT).

**Next steps**:
1. Write `build_dpo_dataset.py` → `data/dpo_dataset/{train,valid}.parquet` + `meta.json`
2. Download `OpenOneRec/OneRec-1.7B`
3. DPO training (trl `DPOTrainer` or repo's RL script) on single A100/A6000
4. Eval on `benchmark_data/video/video_test.parquet` — Recall@10, Pass@32, Pass@1

## Repo layout

```
project/
├── CLAUDE.md                              # this file
├── notebook/
│   ├── EDA.md                             # CURRENT research design + EDA spec (source of truth)
│   ├── EDA_报告.md                         # summary report of all EDA findings (Chinese)
│   ├── eda_data_health.py                 # Script 1: dataset quality checks
│   ├── eda_behavior_signals.py            # Script 2: behavior signal analysis
│   ├── eda_dpo_feasibility.py             # Script 3: DPO pair feasibility
│   ├── eda_ms2.ipynb                      # milestone 2 notebook
│   ├── eda_story.ipynb                    # narrative notebook
│   └── outputs/                           # text reports from EDA scripts
├── data/OpenOneRec/                       # downloaded dataset (not in git)
└── oneRec/*.pdf                           # reference papers (OneRec tech report + dataset notes)
```

EDA scripts each resolve `DATA_DIR` relative to `PROJECT_ROOT` (parent of `notebook/`), pointing to `data/OpenOneRec/`. EDA.md uses `raw_data/` as a placeholder — ignore that; the scripts already use the correct path.

## Dependencies

Python 3.10+. Key packages: `pandas`, `pyarrow`, `numpy`, `huggingface_hub`. For DPO training (upcoming): `torch`, `transformers`, `trl`.

## Commands

Run EDA scripts (each is standalone, run from project root):
```bash
python notebook/eda_data_health.py        # → notebook/outputs/eda_data_health_report.txt
python notebook/eda_behavior_signals.py   # → notebook/outputs/eda_behavior_signals_report.txt
python notebook/eda_dpo_feasibility.py    # → notebook/outputs/eda_dpo_feasibility_report.txt
```

HF auth: token lives at `~/.cache/huggingface/token`. Dataset is gated — request access on the HF repo page first. `huggingface-cli` is not on PATH; use `python -c "from huggingface_hub import login; login()"` if re-auth needed (note: typer-version bug may force you to write the token file by hand).

## Conventions

- All EDA filters to `split=0` only. Never touch `benchmark_data/` for training analysis — it is the held-out test set with 20% of users from the 200K-user pool, completely disjoint from the master table.
- EDA scripts emit text reports to `notebook/outputs/` (or wherever EDA.md specifies); don't dump giant intermediate parquets.
- Three-script structure (data_health → behavior_signals → dpo_feasibility) is a deliberate narrative — keep them independent so each can be re-run in isolation.