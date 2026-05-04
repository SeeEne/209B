"""
build_contrastive_dataset.py  (version 0)

Constructs a contrastive-learning dataset from the OpenOneRec master table.

Positive item : longview=1 OR like=1 OR follow=1 OR forward=1
Negative item : not_interested=1 OR all 5 signals are 0

For each eligible user, randomly sample `prediction_length` positive items
and `prediction_length` negative items from target_video_pid. One pair per user.
All PIDs are mapped to 3-token semantic IDs.

Usage:
    python build_contrastive_dataset.py                  # default length=1
    python build_contrastive_dataset.py --length 3       # length=3

Output:
    data/contrastive_dataset_v1/
        train.parquet   — 90% of pairs
        valid.parquet   — 10% of pairs
        meta.json       — config + counts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "OpenOneRec")
MASTER = os.path.join(DATA_DIR, "onerec_bench_release.parquet")
VIDEO_AD_MAP = os.path.join(DATA_DIR, "video_ad_pid2sid.parquet")

POSITIVE_SIGNALS = ["longview", "like", "follow", "forward"]
NEGATIVE_EXPLICIT = "not_interested"

VALID_FRAC = 0.1
SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_pid2sid(path: str) -> dict[int, list[int]]:
    print(f"Loading PID→SID mapping from {os.path.basename(path)} ...")
    t0 = time.time()
    df = pd.read_parquet(path, columns=["pid", "sid"])
    mapping = {}
    for pid, sid in zip(df["pid"].values, df["sid"].values):
        mapping[int(pid)] = sid.tolist()
    print(f"  {len(mapping):,} entries in {time.time()-t0:.1f}s")
    return mapping


def classify_items(row):
    """Return boolean masks (positive, negative) for target items."""
    pids = row["target_video_pid"]
    if pids is None or len(pids) == 0:
        return None, None

    L = len(pids)
    # Positive: any of longview/like/follow/forward is 1
    pos_mask = np.zeros(L, dtype=bool)
    for sig in POSITIVE_SIGNALS:
        arr = row[f"target_video_{sig}"]
        if arr is not None:
            pos_mask |= (np.asarray(arr) == 1)

    # Negative: not_interested=1 OR all 5 signals are 0
    ni = row[f"target_video_{NEGATIVE_EXPLICIT}"]
    ni_mask = (np.asarray(ni) == 1) if ni is not None else np.zeros(L, dtype=bool)

    all_zero = ~pos_mask & ~ni_mask  # no signal at all
    neg_mask = ni_mask | all_zero

    return pos_mask, neg_mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build contrastive dataset v0")
    parser.add_argument("--length", type=int, default=1,
                        help="Number of positive/negative items per pair (1-5)")
    args = parser.parse_args()
    pred_len = args.length
    assert 1 <= pred_len <= 5, "prediction length must be between 1 and 5"

    out_dir = os.path.join(PROJECT_ROOT, "data", "contrastive_dataset_v1")

    print("=" * 60)
    print(f"Building contrastive dataset v1  (prediction_length={pred_len})")
    print("=" * 60)

    # Load data
    print(f"\nLoading master table ...")
    master = pd.read_parquet(MASTER)
    df = master[master["split"] == 0].reset_index(drop=True)
    print(f"  {len(df):,} split=0 rows")

    pid2sid = build_pid2sid(VIDEO_AD_MAP)

    # Process users
    print(f"\nClassifying target items and sampling pairs ...")
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    records = []
    n_no_hist = 0
    n_empty_target = 0
    n_not_enough_pos = 0
    n_not_enough_neg = 0
    n_sid_missing = 0

    for idx in range(len(df)):
        row = df.iloc[idx]

        # Check history
        hist_pids = row["hist_video_pid"]
        if hist_pids is None or len(hist_pids) == 0:
            n_no_hist += 1
            continue

        # Classify target items
        pos_mask, neg_mask = classify_items(row)
        if pos_mask is None:
            n_empty_target += 1
            continue

        pos_indices = np.where(pos_mask)[0]
        neg_indices = np.where(neg_mask)[0]

        if len(pos_indices) < pred_len:
            n_not_enough_pos += 1
            continue
        if len(neg_indices) < pred_len:
            n_not_enough_neg += 1
            continue

        # Sample
        chosen_idx = rng.choice(pos_indices, size=pred_len, replace=False)
        rejected_idx = rng.choice(neg_indices, size=pred_len, replace=False)

        # Sort by original position (preserve temporal order)
        chosen_idx = np.sort(chosen_idx)
        rejected_idx = np.sort(rejected_idx)

        target_pids = row["target_video_pid"]
        chosen_pids = target_pids[chosen_idx]
        rejected_pids = target_pids[rejected_idx]

        # Convert to SIDs
        hist_sids = [pid2sid[int(p)] for p in hist_pids if int(p) in pid2sid]
        chosen_sids = [pid2sid.get(int(p)) for p in chosen_pids]
        rejected_sids = [pid2sid.get(int(p)) for p in rejected_pids]

        if None in chosen_sids or None in rejected_sids or len(hist_sids) == 0:
            n_sid_missing += 1
            continue

        records.append({
            "uid": int(row["uid"]),
            "hist_sids": hist_sids,
            "chosen_sids": chosen_sids,
            "rejected_sids": rejected_sids,
            "chosen_pids": chosen_pids.tolist(),
            "rejected_pids": rejected_pids.tolist(),
        })

        if (idx + 1) % 40000 == 0:
            print(f"  {idx+1:,} / {len(df):,} users, {len(records):,} pairs so far")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Total pairs        : {len(records):,}")
    print(f"  Skipped (no hist)  : {n_no_hist:,}")
    print(f"  Skipped (empty tgt): {n_empty_target:,}")
    print(f"  Skipped (few pos)  : {n_not_enough_pos:,}")
    print(f"  Skipped (few neg)  : {n_not_enough_neg:,}")
    print(f"  Skipped (SID miss) : {n_sid_missing:,}")

    if len(records) == 0:
        print("ERROR: no pairs produced.")
        sys.exit(1)

    # Train/valid split
    indices = np.arange(len(records))
    rng2 = np.random.default_rng(SEED)
    rng2.shuffle(indices)
    n_valid = int(len(records) * VALID_FRAC)
    valid_set = set(indices[:n_valid].tolist())

    train_records = [records[i] for i in range(len(records)) if i not in valid_set]
    valid_records = [records[i] for i in range(len(records)) if i in valid_set]

    print(f"\n  Train: {len(train_records):,}")
    print(f"  Valid: {len(valid_records):,}")

    # Save
    os.makedirs(out_dir, exist_ok=True)

    for name, recs in [("train", train_records), ("valid", valid_records)]:
        path = os.path.join(out_dir, f"{name}.parquet")
        pd.DataFrame(recs).to_parquet(path, index=False)
        print(f"  Saved {path}")

    meta = {
        "version": "v0",
        "prediction_length": pred_len,
        "positive_signals": POSITIVE_SIGNALS,
        "negative_rule": "not_interested=1 OR all_signals=0",
        "valid_frac": VALID_FRAC,
        "seed": SEED,
        "total_pairs": len(records),
        "train_pairs": len(train_records),
        "valid_pairs": len(valid_records),
        "skipped_no_hist": n_no_hist,
        "skipped_empty_target": n_empty_target,
        "skipped_few_positive": n_not_enough_pos,
        "skipped_few_negative": n_not_enough_neg,
        "skipped_sid_missing": n_sid_missing,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved meta.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
