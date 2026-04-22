"""
Script 2: eda_behavior_signals.py

Analyzes the five user behavior signals on the video task and validates that
they can replace OneRec's Reward Model when constructing DPO preference pairs.

Per notebook/EDA.md §Script 2, we look at:
  1. Per-signal coverage and density
  2. Sparsity ranking across signals
  3. Co-occurrence between signals (e.g. follow ⇒ longview?)
  4. Semantic consistency (within a user, do liked videos have higher
     longview rate than non-liked videos?)

We report on BOTH `hist_video_*` (context side, as EDA.md specifies) AND
`target_video_*` (the side that actually feeds into the DPO reward in
Script 3). The hist side gives a high-density view; the target side tells us
how many DPO pairs each experimental arm can really build.

Outputs: notebook/outputs/eda_behavior_signals_report.txt
"""

from __future__ import annotations

import os
import sys
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "OpenOneRec")
OUT_DIR = os.path.join(PROJECT_ROOT, "notebook", "outputs")
OUT_PATH = os.path.join(OUT_DIR, "eda_behavior_signals_report.txt")
MASTER = os.path.join(DATA_DIR, "onerec_bench_release.parquet")

SIGNALS = ["longview", "like", "follow", "forward", "not_interested"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def to_int_array(arr) -> np.ndarray:
    """Convert a list-cell to a 1-D int8 numpy array, dropping None/NaN."""
    if arr is None:
        return np.empty(0, dtype=np.int8)
    a = np.asarray(arr)
    if a.dtype.kind == "f":
        a = a[~np.isnan(a)]
    return a.astype(np.int8, copy=False)


def per_user_signal_counts(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """For each row return a dataframe of {signal: positive count, len: pid len}."""
    pid_lens = df[f"{prefix}_pid"].apply(lambda a: 0 if a is None else len(a)).values
    out = {"len": pid_lens}
    for sig in SIGNALS:
        col = f"{prefix}_{sig}"
        out[sig] = df[col].apply(lambda a: int(to_int_array(a).sum())).values
    return pd.DataFrame(out, index=df.index)


def fmt_pct(x: float) -> str:
    return f"{x:7.2%}"


# ---------------------------------------------------------------------------
# Section 1 — coverage / density per signal
# ---------------------------------------------------------------------------

def report_coverage(counts: pd.DataFrame, label: str) -> None:
    section(f"1. Per-signal coverage & density — {label}")
    n_users = len(counts)
    n_nonempty = int((counts["len"] > 0).sum())
    print(f"users (rows)          : {n_users:,}")
    print(f"users with non-empty {label}_pid: {n_nonempty:,}  ({n_nonempty/n_users:.2%})")
    print()
    print(f"  {'signal':16s} {'users_with_signal':>20s} {'coverage':>12s} "
          f"{'mean/user':>12s} {'p50':>6s} {'p95':>6s} {'max':>6s}")
    rows = []
    for sig in SIGNALS:
        col = counts[sig]
        users_with = int((col > 0).sum())
        cov = users_with / n_users
        rows.append({
            "signal": sig,
            "users_with_signal": users_with,
            "coverage": cov,
            "mean": col.mean(),
            "p50": col.quantile(0.5),
            "p95": col.quantile(0.95),
            "max": int(col.max()),
        })
    rows.sort(key=lambda r: -r["coverage"])
    for r in rows:
        print(f"  {r['signal']:16s} {r['users_with_signal']:>20,} {fmt_pct(r['coverage']):>12s} "
              f"{r['mean']:>12.2f} {r['p50']:>6.0f} {r['p95']:>6.0f} {r['max']:>6d}")
    print()
    print("Sparsity ranking (most → least common):")
    print("  " + " > ".join(r["signal"] for r in rows))


# ---------------------------------------------------------------------------
# Section 2 — item-level positive rates
# ---------------------------------------------------------------------------

def report_item_level_rates(df: pd.DataFrame, prefix: str, label: str) -> None:
    section(f"2. Item-level positive rates — {label}")
    print("Across ALL items in ALL users (not per user — per impression).")
    print()
    totals = {}
    print(f"  {'signal':16s} {'positives':>14s} {'total_items':>14s} {'pos_rate':>12s}")
    for sig in SIGNALS:
        col = df[f"{prefix}_{sig}"]
        pos = 0
        tot = 0
        for arr in col.values:
            a = to_int_array(arr)
            tot += a.size
            pos += int((a == 1).sum())
        totals[sig] = (pos, tot)
        rate = pos / tot if tot else 0
        print(f"  {sig:16s} {pos:>14,} {tot:>14,} {fmt_pct(rate):>12s}")


# ---------------------------------------------------------------------------
# Section 3 — co-occurrence between signals (per user)
# ---------------------------------------------------------------------------

def report_cooccurrence(counts: pd.DataFrame, label: str) -> None:
    section(f"3. Signal co-occurrence (per-user) — {label}")
    print("Cell (row, col) = P(user has col-signal | user has row-signal).")
    print("Diagonal = 1.0 by construction.\n")

    has = {sig: (counts[sig] > 0).values for sig in SIGNALS}
    print(f"  {'given ↓':16s}" + "".join(f"{s:>14s}" for s in SIGNALS))
    for r in SIGNALS:
        denom = int(has[r].sum())
        line = f"  {r:16s}"
        for c in SIGNALS:
            if denom == 0:
                line += f"{'-':>14s}"
            else:
                num = int((has[r] & has[c]).sum())
                line += f"{num/denom:>14.2%}"
        line += f"   (n={denom:,})"
        print(line)


# ---------------------------------------------------------------------------
# Section 4 — semantic consistency
# ---------------------------------------------------------------------------

def report_semantic_consistency(df: pd.DataFrame, prefix: str, label: str) -> None:
    section(f"4. Semantic consistency — {label}")
    print("Hypothesis: explicit signals (like / follow) should imply higher")
    print("longview rate than the unconditional baseline. We compute, across")
    print("ALL items in ALL users:")
    print("  P(longview=1 | <signal>=1)   vs   P(longview=1 | <signal>=0)")
    print("Lift > 1 supports the signal as a stronger preference indicator.\n")

    lv_col = f"{prefix}_longview"
    print(f"  {'condition':22s} {'P(longview|sig=1)':>22s} "
          f"{'P(longview|sig=0)':>22s} {'lift':>8s} {'n_items':>14s}")
    for sig in SIGNALS:
        if sig == "longview":
            continue
        sc = f"{prefix}_{sig}"
        pos_lv = pos_n = neg_lv = neg_n = 0
        for lv_arr, sg_arr in zip(df[lv_col].values, df[sc].values):
            lv = to_int_array(lv_arr)
            sg = to_int_array(sg_arr)
            if lv.size != sg.size or lv.size == 0:
                continue
            mask_pos = sg == 1
            mask_neg = ~mask_pos
            pos_n += int(mask_pos.sum())
            pos_lv += int(lv[mask_pos].sum())
            neg_n += int(mask_neg.sum())
            neg_lv += int(lv[mask_neg].sum())
        p_pos = pos_lv / pos_n if pos_n else 0
        p_neg = neg_lv / neg_n if neg_n else 0
        lift = (p_pos / p_neg) if p_neg else float("inf")
        print(f"  {sig:22s} {fmt_pct(p_pos):>22s} {fmt_pct(p_neg):>22s} "
              f"{lift:>8.2f} {pos_n+neg_n:>14,}")


# ---------------------------------------------------------------------------
# Section 5 — DPO target-side feasibility preview
# ---------------------------------------------------------------------------

def report_target_pair_feasibility(target_counts: pd.DataFrame) -> None:
    section("5. Target-side DPO pair feasibility preview")
    print("For each experimental arm we count users whose target_video sequence")
    print("contains the signals required to build a (chosen, rejected) pair.")
    print("Pair feasibility = arm needs at least 1 positive AND 1 negative item")
    print("in the target sequence.\n")

    valid = target_counts[target_counts["len"] > 0]
    n = len(valid)
    print(f"Users with non-empty target_video_pid: {n:,}\n")

    # Each arm: (name, positive signals, negative signals, condition function)
    def cond_arm1(row):  # longview only — need at least 1 longview AND 1 non-longview
        return row["longview"] >= 1 and (row["len"] - row["longview"]) >= 1

    def cond_arm2(row):  # implicit + explicit positive
        pos = max(row["longview"], row["like"], row["follow"])
        return (row["longview"] + row["like"] + row["follow"]) >= 1 and (row["len"] - row["longview"]) >= 1 and pos >= 1

    def cond_arm3(row):  # full signals — needs pos AND not_interested
        return (row["longview"] + row["like"] + row["follow"]) >= 1 and row["not_interested"] >= 1

    def cond_arm4(row):  # pure explicit
        return (row["like"] + row["follow"]) >= 1 and (row["not_interested"] >= 1 or row["len"] - row["like"] - row["follow"] >= 1)

    arms = [
        ("Arm 1 — longview only",          cond_arm1),
        ("Arm 2 — longview+like+follow",   cond_arm2),
        ("Arm 3 — full signals (needs neg)", cond_arm3),
        ("Arm 4 — pure explicit",          cond_arm4),
    ]

    print(f"  {'arm':40s} {'feasible_users':>16s} {'share':>10s}")
    for name, fn in arms:
        ok = int(valid.apply(fn, axis=1).sum())
        print(f"  {name:40s} {ok:>16,} {ok/n:>10.2%}")
    print()
    print("Note: this is a per-user feasibility check, not pair count. Script 3")
    print("will compute the actual pair budget under each arm's reward formula.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    buf = StringIO()
    with redirect_stdout(buf):
        print("OpenOneRec — Script 2: Behavior Signals Report")
        print(f"Generated by {os.path.basename(__file__)}")
        print(f"Data dir : {DATA_DIR}")

        df = pd.read_parquet(MASTER)
        df = df[df["split"] == 0].reset_index(drop=True)
        print(f"Loaded {len(df):,} split=0 rows")

        # ---- HIST side ----
        hist_counts = per_user_signal_counts(df, "hist_video")
        report_coverage(hist_counts, "hist_video_*")
        report_item_level_rates(df, "hist_video", "hist_video_*")
        report_cooccurrence(hist_counts, "hist_video_*")
        report_semantic_consistency(df, "hist_video", "hist_video_*")

        # ---- TARGET side (this is what DPO actually consumes) ----
        target_counts = per_user_signal_counts(df, "target_video")
        report_coverage(target_counts, "target_video_*")
        report_item_level_rates(df, "target_video", "target_video_*")
        report_cooccurrence(target_counts, "target_video_*")
        report_semantic_consistency(df, "target_video", "target_video_*")
        report_target_pair_feasibility(target_counts)

        section("Summary")
        print("Use this section to populate the EDA narrative.")
        print()
        print("Hist side gives high-density signal stats (~484 items/user).")
        print("Target side determines DPO pair budget (only ~9 items/user) and")
        print("is the binding constraint for our four experimental arms.")
        print()
        print("Done.")

    text = buf.getvalue()
    with open(OUT_PATH, "w") as f:
        f.write(text)
    sys.stdout.write(text)
    sys.stdout.write(f"\n[wrote report to {OUT_PATH}]\n")


if __name__ == "__main__":
    main()