"""
Script 1: eda_data_health.py

Verifies basic dataset quality so we know the OpenOneRec subset is usable for
DPO experiments. Covers master table health, sequence-length distributions,
null rates, behavior-label/pid length alignment (a hard invariant for the
reward computation in Script 3), mapping-table sanity, and PID coverage.

Outputs a single text report at outputs/eda_data_health_report.txt.

Scope (per notebook/EDA.md):
- video task only (hist_video_* / target_video_pid)
- split=0 only (the subset SFT scripts actually consume)
- pid2caption is NOT downloaded in the minimal subset; that section is skipped
  with a clear note in the report.
"""

from __future__ import annotations

import os
import sys
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "OpenOneRec")
OUT_DIR = os.path.join(PROJECT_ROOT, "notebook", "outputs")
OUT_PATH = os.path.join(OUT_DIR, "eda_data_health_report.txt")

MASTER = os.path.join(DATA_DIR, "onerec_bench_release.parquet")
VIDEO_AD_MAP = os.path.join(DATA_DIR, "video_ad_pid2sid.parquet")
PRODUCT_MAP = os.path.join(DATA_DIR, "product_pid2sid.parquet")
PID2CAPTION = os.path.join(DATA_DIR, "pid2caption.parquet")  # not in minimal subset

VIDEO_BEHAVIOR_COLS = [
    "hist_video_longview",
    "hist_video_like",
    "hist_video_follow",
    "hist_video_forward",
    "hist_video_not_interested",
]
TARGET_VIDEO_BEHAVIOR_COLS = [
    "target_video_longview",
    "target_video_like",
    "target_video_follow",
    "target_video_forward",
    "target_video_not_interested",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def is_list_col(series: pd.Series) -> bool:
    """Detect a list-of-values column by sampling non-null values, not iloc[0]."""
    sample = series.dropna()
    if len(sample) == 0:
        return False
    sample = sample.head(20)
    return all(isinstance(v, (list, np.ndarray)) for v in sample)


def list_len_valid(arr) -> int:
    """Length of a list, treating internal NaNs as invalid (not counted)."""
    if arr is None:
        return 0
    n = 0
    for v in arr:
        if pd.isna(v):
            continue
        n += 1
    return n


def list_len_raw(arr) -> int:
    """Raw length including NaN padding."""
    return 0 if arr is None else len(arr)


def length_stats(lens: pd.Series) -> dict:
    return {
        "mean": float(lens.mean()),
        "median": float(lens.median()),
        "p25": float(lens.quantile(0.25)),
        "p75": float(lens.quantile(0.75)),
        "p95": float(lens.quantile(0.95)),
        "p99": float(lens.quantile(0.99)),
        "max": int(lens.max()),
        "empty_rate": float((lens == 0).mean()),
    }


def fmt_stats(stats: dict) -> str:
    return (
        f"mean={stats['mean']:8.2f}  median={stats['median']:6.0f}  "
        f"p25={stats['p25']:6.0f}  p75={stats['p75']:6.0f}  "
        f"p95={stats['p95']:6.0f}  p99={stats['p99']:6.0f}  "
        f"max={stats['max']:6d}  empty={stats['empty_rate']:.2%}"
    )


def hist_buckets(lens: pd.Series, edges: list[int]) -> str:
    counts = []
    prev = 0
    for e in edges:
        c = int(((lens >= prev) & (lens < e)).sum())
        counts.append((f"[{prev:4d},{e:5d})", c))
        prev = e
    counts.append((f"[{prev:4d},  inf)", int((lens >= prev).sum())))
    total = sum(c for _, c in counts)
    lines = []
    for label, c in counts:
        pct = c / total if total else 0
        bar = "#" * int(pct * 50)
        lines.append(f"  {label}  {c:>10,}  {pct:6.2%}  {bar}")
    return "\n".join(lines)


def flatten_pids(series: pd.Series) -> np.ndarray:
    """Concatenate all list elements into a single int64 numpy array,
    dropping None and NaN. Vectorized to avoid Python double-loop."""
    parts = []
    for arr in series.values:
        if arr is None:
            continue
        a = np.asarray(arr)
        if a.size == 0:
            continue
        if a.dtype.kind == "f":
            a = a[~np.isnan(a)]
        if a.size:
            parts.append(a.astype(np.int64, copy=False))
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def report_master_basics(master: pd.DataFrame) -> pd.DataFrame:
    section("1. Master table — basic info")
    print(f"file       : {MASTER}")
    print(f"total rows : {len(master):,}")
    print(f"unique uids: {master['uid'].nunique():,}")
    print(f"columns    : {len(master.columns)}")
    print()
    print("dtypes (parquet schema):")
    schema = pq.ParquetFile(MASTER).schema_arrow
    for field in schema:
        print(f"  {field.name:35s} {field.type}")
    print()
    print("split distribution:")
    sc = master["split"].value_counts().sort_index()
    for k, v in sc.items():
        print(f"  split={k}: {v:>10,}  ({v/len(master):.2%})")

    train = master[master["split"] == 0].reset_index(drop=True)
    print()
    print(f">>> Using split=0 subset for the rest of this report: {len(train):,} rows")
    return train


def report_sequence_lengths(train: pd.DataFrame) -> None:
    section("2. Sequence length distributions (split=0)")
    print("Reporting BOTH raw length (including NaN padding) and valid length")
    print("(NaN-stripped). For int64 list columns these match; for double list")
    print("columns (hist_ad_pid, hist_goods_pid, hist_longview_video_list) the")
    print("gap reveals NaN padding.")

    cols = [
        "hist_video_pid",
        "target_video_pid",
        "hist_ad_pid",
        "hist_goods_pid",
        "hist_longview_video_list",
        "target_ad_pid",
        "target_goods_pid",
    ]
    edges = [1, 5, 10, 20, 50, 100, 200, 500]

    for c in cols:
        if c not in train.columns:
            continue
        raw = train[c].apply(list_len_raw)
        valid = train[c].apply(list_len_valid)
        nan_share = (raw.sum() - valid.sum()) / max(raw.sum(), 1)
        print(f"\n[{c}]")
        print("  raw   " + fmt_stats(length_stats(raw)))
        print("  valid " + fmt_stats(length_stats(valid)))
        print(f"  internal NaN share: {nan_share:.2%}")
        if c in ("hist_video_pid", "target_video_pid"):
            print("  bucket distribution (valid length):")
            print(hist_buckets(valid, edges))
            # Report P95 within non-empty users only — used for K truncation.
            non_empty = valid[valid > 0]
            if len(non_empty):
                print(f"  P95 (non-empty users only): {non_empty.quantile(.95):.0f}")


def report_field_nulls(train: pd.DataFrame) -> None:
    section("3. Field null / empty rates (split=0)")

    list_cols, str_cols, scalar_cols = [], [], []
    for c in train.columns:
        s = train[c]
        if pd.api.types.is_object_dtype(s):
            if is_list_col(s):
                list_cols.append(c)
            else:
                str_cols.append(c)
        else:
            scalar_cols.append(c)

    print("\nList columns (None or empty list):")
    print(f"  {'column':35s} {'null':>8s} {'null+empty':>12s}")
    rows = []
    for c in list_cols:
        s = train[c]
        none_rate = float(s.isna().mean())
        empty_rate = float(s.apply(lambda x: x is None or len(x) == 0).mean())
        rows.append((c, none_rate, empty_rate))
    rows.sort(key=lambda r: -r[2])
    for c, n, e in rows:
        print(f"  {c:35s} {n:8.2%} {e:12.2%}")

    print("\nString columns (None or empty string):")
    print(f"  {'column':35s} {'null':>8s} {'null+empty':>12s}")
    rows = []
    for c in str_cols:
        s = train[c]
        none_rate = float(s.isna().mean())
        empty_rate = float(s.apply(lambda x: x is None or (isinstance(x, str) and len(x) == 0)).mean())
        rows.append((c, none_rate, empty_rate))
    rows.sort(key=lambda r: -r[2])
    for c, n, e in rows:
        print(f"  {c:35s} {n:8.2%} {e:12.2%}")

    print("\nScalar columns:")
    print(f"  {'column':35s} {'null':>8s}")
    for c in scalar_cols:
        print(f"  {c:35s} {train[c].isna().mean():8.2%}")


def report_alignment_invariants(train: pd.DataFrame) -> None:
    section("4. Behavior label / pid length alignment (HARD invariant)")
    print("Each hist_video_<behavior> list MUST have the same length as")
    print("hist_video_pid for the same row — otherwise reward computation in")
    print("Script 3 will misalign behaviors with items.\n")

    def check(pid_col: str, behavior_cols: list[str]) -> None:
        pid_lens = train[pid_col].apply(list_len_raw).values
        print(f"[{pid_col}]")
        for bc in behavior_cols:
            if bc not in train.columns:
                print(f"  {bc:35s} MISSING")
                continue
            blens = train[bc].apply(list_len_raw).values
            mismatched = int((pid_lens != blens).sum())
            print(f"  {bc:35s} mismatched_rows={mismatched:>8,}  "
                  f"({mismatched/len(train):.4%})")
        print()

    check("hist_video_pid", VIDEO_BEHAVIOR_COLS)
    check("target_video_pid", TARGET_VIDEO_BEHAVIOR_COLS)


def report_mapping_table(name: str, path: str) -> set[int]:
    print(f"\n[{name}]  file: {path}")
    df = pd.read_parquet(path)
    n = len(df)
    n_unique = df["pid"].nunique()
    sid_lens = df["sid"].apply(len)
    print(f"  rows           : {n:,}")
    print(f"  unique pids    : {n_unique:,}")
    print(f"  one-to-one     : {n == n_unique}")
    print(f"  sid length ==3 : {(sid_lens == 3).all()}  (min={sid_lens.min()}, max={sid_lens.max()})")
    if not (sid_lens == 3).all():
        print("  sid length value counts:")
        for k, v in sid_lens.value_counts().sort_index().items():
            print(f"    len={k}: {v:,}")
    return set(df["pid"].astype("int64").values)


def report_mappings() -> tuple[set[int], set[int]]:
    section("5. Mapping table health")
    video_ad = report_mapping_table("video_ad_pid2sid", VIDEO_AD_MAP)
    product = report_mapping_table("product_pid2sid", PRODUCT_MAP)

    print()
    if os.path.exists(PID2CAPTION):
        print("[pid2caption] downloaded — extending report.")
        df = pd.read_parquet(PID2CAPTION)
        print(f"  rows: {len(df):,}  cols: {list(df.columns)}")
        text_col = next((c for c in ("caption", "dense_caption") if c in df.columns), None)
        if text_col is None:
            print("  WARNING: neither 'caption' nor 'dense_caption' present.")
        else:
            print(f"  text field detected: {text_col}")
            lens = df[text_col].astype(str).str.len()
            print(f"  text length: mean={lens.mean():.1f} p50={lens.quantile(.5):.0f} "
                  f"p95={lens.quantile(.95):.0f} max={lens.max()}")
            print(f"  empty caption rate: {(df[text_col].isna() | (lens == 0)).mean():.2%}")
    else:
        print("[pid2caption] NOT downloaded in the minimal viable subset — skipping.")
        print("  (Re-add 'pid2caption.parquet' to ALLOW_PATTERNS in old/onerec_collector.py if needed.)")

    return video_ad, product


def coverage_vectorized(series: pd.Series, ref_pids: np.ndarray) -> tuple[int, int, float]:
    """Vectorized PID coverage: flatten the list column once, isin against
    a sorted numpy array of reference pids."""
    flat = flatten_pids(series)
    if flat.size == 0:
        return 0, 0, 0.0
    hit = int(np.isin(flat, ref_pids, assume_unique=False).sum())
    return hit, int(flat.size), hit / flat.size


def report_pid_coverage(train: pd.DataFrame, video_ad: set, product: set) -> None:
    section("6. PID coverage in mapping tables (split=0)")
    print("Any pid not present in the mapping will be silently dropped by SFT scripts.\n")

    # Convert reference sets to sorted numpy arrays for vectorized isin.
    video_ad_arr = np.fromiter(video_ad, dtype=np.int64, count=len(video_ad))
    video_ad_arr.sort()
    product_arr = np.fromiter(product, dtype=np.int64, count=len(product))
    product_arr.sort()

    checks = [
        ("hist_video_pid", video_ad_arr),
        ("target_video_pid", video_ad_arr),
        ("hist_ad_pid", video_ad_arr),
        ("target_ad_pid", video_ad_arr),
        ("hist_goods_pid", product_arr),
        ("target_goods_pid", product_arr),
    ]
    print(f"  {'column':22s} {'pids_seen':>14s} {'pids_in_map':>14s} {'coverage':>10s}")
    for col, ref in checks:
        hit, seen, rate = coverage_vectorized(train[col], ref)
        print(f"  {col:22s} {seen:>14,} {hit:>14,} {rate:>10.4%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    buf = StringIO()
    with redirect_stdout(buf):
        print("OpenOneRec — Script 1: Data Health Report")
        print(f"Generated by {os.path.basename(__file__)}")
        print(f"Data dir : {DATA_DIR}")

        master = pd.read_parquet(MASTER)
        train = report_master_basics(master)
        report_sequence_lengths(train)
        report_field_nulls(train)
        report_alignment_invariants(train)
        video_ad, product = report_mappings()
        report_pid_coverage(train, video_ad, product)

        section("Summary")
        hv = train["hist_video_pid"].apply(list_len_valid)
        tv = train["target_video_pid"].apply(list_len_valid)
        hv_ne = hv[hv > 0]
        print("Key numbers to copy into the EDA narrative:")
        print(f"  - split=0 row count            : {len(train):,}")
        print(f"  - hist_video_pid  P50/P95 (all): {hv.quantile(.5):.0f} / {hv.quantile(.95):.0f}")
        if len(hv_ne):
            print(f"  - hist_video_pid  P95 (non-empty): {hv_ne.quantile(.95):.0f}")
        print(f"  - target_video_pid P50/P95     : {tv.quantile(.5):.0f} / {tv.quantile(.95):.0f}")
        print(f"  - video_ad map pids            : {len(video_ad):,}")
        print(f"  - product map pids             : {len(product):,}")
        print()
        print("Done.")

    text = buf.getvalue()
    with open(OUT_PATH, "w") as f:
        f.write(text)
    sys.stdout.write(text)
    sys.stdout.write(f"\n[wrote report to {OUT_PATH}]\n")


if __name__ == "__main__":
    main()