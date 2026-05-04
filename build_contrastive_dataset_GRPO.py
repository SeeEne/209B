"""
build_contrastive_dataset_GRPO.py

Expand contrastive_dataset_v1 (length=3, 3 chosen + 3 rejected per user)
into a GRPO-style dataset where each user contributes G single-item
contrastive pairs that share a group_id.

Pairing order (PAIRING_ORDER): we pre-rank all 9 (chosen_i, rejected_j)
combinations from the 3x3 grid:

    PAIRING_ORDER[0..2]  = cyclic shift 1 derangement
                           (0,1), (1,2), (2,0)        — each chosen & rejected used 1×
    PAIRING_ORDER[3..5]  = cyclic shift 2 derangement
                           (0,2), (1,0), (2,1)        — each chosen & rejected used 1×
    PAIRING_ORDER[6..8]  = diagonal (matched positions)
                           (0,0), (1,1), (2,2)        — chosen[i] paired with rejected[i]

For G ∈ [1, 9], take PAIRING_ORDER[:G].

Each output row =
    {
        group_id,       # unique per source-row (== one user / session)
        uid,
        g,              # 0..G-1, position within group
        hist_sids,      # same across all G rows of one group
        chosen_sid,     # single [a, b, c]
        rejected_sid,   # single [a, b, c]
        chosen_pid,     # single int (preserved for engagement eval)
        rejected_pid,
    }

Total output rows = G × input rows. group_id ranges over [0, len(input)).

Usage:
    python build_contrastive_dataset_GRPO.py                 # default G=3
    python build_contrastive_dataset_GRPO.py --G 5
    python build_contrastive_dataset_GRPO.py --G 9 \
        --output_dir data/contrastive_dataset_v1_grpo_g9

Output dir defaults to data/contrastive_dataset_v1_grpo (G=3) or
data/contrastive_dataset_v1_grpo_g{G} for G != 3.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "contrastive_dataset_v1"

# 9 (chosen_i, rejected_j) pairs from the 3x3 grid, in priority order:
#   0..2: cyclic shift 1  (derangement)
#   3..5: cyclic shift 2  (derangement)
#   6..8: diagonal        (matched positions)
# G=3 uses the first 3 (cyclic shift 1 — backwards compatible with v1_grpo);
# G=6 covers both derangements; G=9 is the full grid.
PAIRING_ORDER = [
    (0, 1), (1, 2), (2, 0),    # G in [1, 3] → cyclic shift 1 only
    (0, 2), (1, 0), (2, 1),    # G in [4, 6] → + cyclic shift 2
    (0, 0), (1, 1), (2, 2),    # G in [7, 9] → + diagonal
]
MAX_G = len(PAIRING_ORDER)     # 9


def default_output_dir(G: int) -> Path:
    if G == 3:
        return PROJECT_ROOT / "data" / "contrastive_dataset_v1_grpo"
    return PROJECT_ROOT / "data" / f"contrastive_dataset_v1_grpo_g{G}"


def expand_split(df: pd.DataFrame, G: int) -> pd.DataFrame:
    """Each input row -> G output rows, using PAIRING_ORDER[:G]."""
    has_pids = "chosen_pids" in df.columns and "rejected_pids" in df.columns
    pairs = PAIRING_ORDER[:G]

    out_rows = []
    for src_idx, row in tqdm(df.iterrows(), total=len(df),
                             desc=f"Expanding G={G}"):
        chosen_sids = row["chosen_sids"]
        rejected_sids = row["rejected_sids"]
        assert len(chosen_sids) == 3, (
            f"row {src_idx}: expected 3 chosen items, got {len(chosen_sids)}. "
            f"Did you build v1 with --length 3?"
        )
        assert len(rejected_sids) == 3, (
            f"row {src_idx}: expected 3 rejected items, got {len(rejected_sids)}"
        )

        chosen_pids = row["chosen_pids"] if has_pids else [None, None, None]
        rejected_pids = row["rejected_pids"] if has_pids else [None, None, None]
        uid = int(row["uid"])
        hist_sids = row["hist_sids"]

        for g, (cidx, ridx) in enumerate(pairs):
            out_rows.append({
                "group_id": int(src_idx),
                "uid": uid,
                "g": g,
                "hist_sids": hist_sids,
                "chosen_sid": chosen_sids[cidx],
                "rejected_sid": rejected_sids[ridx],
                "chosen_pid": (int(chosen_pids[cidx])
                               if chosen_pids[cidx] is not None else None),
                "rejected_pid": (int(rejected_pids[ridx])
                                 if rejected_pids[ridx] is not None else None),
            })
    return pd.DataFrame(out_rows)


def describe_pairing(G: int) -> str:
    pairs = PAIRING_ORDER[:G]
    chosen_counts = [0, 0, 0]
    rejected_counts = [0, 0, 0]
    for c, r in pairs:
        chosen_counts[c] += 1
        rejected_counts[r] += 1
    return (
        f"pairs = {pairs}\n"
        f"  chosen[i] usage: {chosen_counts}  (target: ~{G/3:.2f} each)\n"
        f"  rejected[j] usage: {rejected_counts}  (target: ~{G/3:.2f} each)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Expand v1 (length=3) into GRPO-style single-pair rows."
    )
    parser.add_argument("--G", type=int, default=3,
                        help=f"Pairs per group, in [1, {MAX_G}]. "
                             "G=3 = cyclic shift 1 (default). G=6 = both "
                             "derangements. G=9 = full 3x3 grid.")
    parser.add_argument("--input_dir", default=str(DEFAULT_INPUT_DIR),
                        help="Directory with v1 train.parquet / valid.parquet")
    parser.add_argument("--output_dir", default=None,
                        help="Where to write expanded train/valid parquet. "
                             "Default: data/contrastive_dataset_v1_grpo (G=3) "
                             "or data/contrastive_dataset_v1_grpo_g{G}.")
    args = parser.parse_args()

    assert 1 <= args.G <= MAX_G, f"--G must be in [1, {MAX_G}]"

    G = args.G
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir(G)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Building GRPO dataset (G={G})")
    print(f"  source : {in_dir}")
    print(f"  target : {out_dir}")
    print("=" * 60)
    print(describe_pairing(G))
    print("=" * 60)

    summary = {}
    for split in ("train", "valid"):
        in_path = in_dir / f"{split}.parquet"
        if not in_path.exists():
            print(f"\n[skip] {split}: {in_path} not found")
            continue

        print(f"\n=== {split} ===")
        df = pd.read_parquet(in_path).reset_index(drop=True)
        print(f"  source rows : {len(df):,}")

        expanded = expand_split(df, G)
        print(f"  output rows : {len(expanded):,}  ({G}× source)")
        print(f"  unique groups: {expanded['group_id'].nunique():,}")

        out_path = out_dir / f"{split}.parquet"
        expanded.to_parquet(out_path, index=False)
        print(f"  saved → {out_path}")

        summary[split] = {
            "source_rows": int(len(df)),
            "output_rows": int(len(expanded)),
            "groups": int(expanded["group_id"].nunique()),
        }

    meta = {
        "version": f"v1_grpo_g{G}",
        "source_version": "v1",
        "G": G,
        "pairing_order": [list(p) for p in PAIRING_ORDER[:G]],
        "pairing_explanation": (
            "PAIRING_ORDER ranks the 9 (chosen_i, rejected_j) combinations: "
            "first 3 are cyclic-shift-1 derangement, next 3 are cyclic-shift-2 "
            "derangement, last 3 are the diagonal. Take PAIRING_ORDER[:G]."
        ),
        "row_format": {
            "group_id": "int, unique per source-row (== one user/session)",
            "uid": "int, user id (preserved)",
            "g": f"int in [0, {G}), position within group",
            "hist_sids": "list[[a,b,c]], shared across all G rows of a group",
            "chosen_sid": "[a, b, c], single SID",
            "rejected_sid": "[a, b, c], single SID",
            "chosen_pid": "int or None",
            "rejected_pid": "int or None",
        },
        "training_note": (
            "GRPO trainer must keep all G rows of a group_id in the same "
            "micro-batch for std normalization. Pass --G to the trainer "
            "and ensure --per_device_batch_size is a multiple of G."
        ),
        "splits": summary,
    }
    src_meta = in_dir / "meta.json"
    if src_meta.exists():
        with open(src_meta) as f:
            meta["source_meta"] = json.load(f)
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved meta.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
