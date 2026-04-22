"""
Script 3: eda_dpo_feasibility.py

Given the four DPO experimental arms, compute per-user (chosen, rejected)
sub-sessions inside `target_video_pid` and analyze whether the resulting
preference pairs have enough signal to train DPO.

Per-arm reward formula (per item; session score = sum of item scores):

  Arm 1  longview only            : 1.0*lv
  Arm 2  longview + like + follow : 1.0*lv + 1.5*like + 2.0*follow + 1.5*forward
  Arm 3  Arm 2 + neg penalty      : Arm 2 - 2.0*not_interested
  Arm 4  pure explicit            :        1.5*like + 2.0*follow + 1.5*forward - 2.0*not_interested

Pair construction strategy (default):
  - For a user with target length L and chosen sub-session size m = L // 2,
    rank items by per-item score under each arm; chosen = top-m items,
    rejected = bottom-m items. Sum of top-m by item score equals the maximum
    sum over all m-subsets, so we don't need to enumerate C(L, m) subsets.
  - Users with L < 2*m_min (default m_min=4, so L<8) are dropped.
  - Within the same user, the same m is used across all arms so subsets are
    directly comparable.

This is the "fix" for the original Arm 3 problem: we no longer require an
explicit not_interested sample to construct a pair. not_interested only
contributes to the reward as a soft penalty when present.

Outputs: notebook/outputs/eda_dpo_feasibility_report.txt
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
OUT_PATH = os.path.join(OUT_DIR, "eda_dpo_feasibility_report.txt")
MASTER = os.path.join(DATA_DIR, "onerec_bench_release.parquet")

# Reward weights (item-level). Edit here to ablate weighting.
ARMS = {
    "Arm1_longview":  {"longview": 1.0, "like": 0.0, "follow": 0.0, "forward": 0.0, "not_interested":  0.0},
    "Arm2_implicit_explicit_pos": {"longview": 1.0, "like": 1.5, "follow": 2.0, "forward": 1.5, "not_interested":  0.0},
    "Arm3_full":      {"longview": 1.0, "like": 1.5, "follow": 2.0, "forward": 1.5, "not_interested": -2.0},
    "Arm4_explicit_only": {"longview": 0.0, "like": 1.5, "follow": 2.0, "forward": 1.5, "not_interested": -2.0},
}
SIGNALS = ["longview", "like", "follow", "forward", "not_interested"]
GAP_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
MIN_HALF = 4   # require m >= 4 → target length >= 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def to_int_array(arr) -> np.ndarray:
    if arr is None:
        return np.empty(0, dtype=np.int8)
    a = np.asarray(arr)
    if a.dtype.kind == "f":
        a = a[~np.isnan(a)]
    return a.astype(np.int8, copy=False)


def build_signal_matrix(df: pd.DataFrame, prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For variable-length targets, return:
        sig: (n_items_total, 5) int8 matrix of behavior signals
        offsets: (n_users+1,) int64 offsets so user u has rows offsets[u]:offsets[u+1]
        lengths: (n_users,) int64 length of each user's target
    Items with hist length 0 are kept as empty rows (offsets equal).
    """
    pid_col = df[f"{prefix}_pid"].values
    sig_cols = [df[f"{prefix}_{s}"].values for s in SIGNALS]
    n_users = len(df)
    lengths = np.zeros(n_users, dtype=np.int64)
    parts = [[] for _ in range(5)]
    for i in range(n_users):
        pid = pid_col[i]
        L = 0 if pid is None else len(pid)
        lengths[i] = L
        if L == 0:
            continue
        for k, sig_arr_col in enumerate(sig_cols):
            arr = sig_arr_col[i]
            a = np.asarray(arr) if arr is not None else np.empty(0)
            if a.size != L:
                # Should never happen — Script 1 verified zero mismatch.
                a = np.zeros(L, dtype=np.int8)
            parts[k].append(a.astype(np.int8, copy=False))
    cat = [np.concatenate(p) if p else np.empty(0, dtype=np.int8) for p in parts]
    sig = np.stack(cat, axis=1)  # (total_items, 5)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return sig, offsets, lengths


def per_arm_chosen_rejected(sig: np.ndarray, offsets: np.ndarray, lengths: np.ndarray,
                            weights: dict, m_min: int) -> dict:
    """For each user with length >= 2*m_min, compute:
        m = length // 2
        item_scores = sig @ w
        chosen_items  = indices of top m item_scores
        rejected_items = indices of bottom m item_scores
        chosen_score = sum of top m item_scores
        rejected_score = sum of bottom m item_scores
    Returns dict of arrays keyed by user index (subset of users that qualify).
    """
    w = np.array([weights[s] for s in SIGNALS], dtype=np.float32)
    item_scores_all = sig.astype(np.float32) @ w  # (total_items,)

    n_users = len(lengths)
    elig = np.where(lengths >= 2 * m_min)[0]

    chosen_score = np.full(n_users, np.nan, dtype=np.float32)
    rejected_score = np.full(n_users, np.nan, dtype=np.float32)
    m_used = np.zeros(n_users, dtype=np.int16)
    chosen_items = [None] * n_users  # list of np arrays
    rejected_items = [None] * n_users

    for u in elig:
        L = int(lengths[u])
        m = L // 2
        s = item_scores_all[offsets[u]:offsets[u + 1]]
        order = np.argsort(s, kind="stable")  # ascending
        bot = order[:m]
        top = order[-m:]
        chosen_score[u] = float(s[top].sum())
        rejected_score[u] = float(s[bot].sum())
        m_used[u] = m
        chosen_items[u] = top
        rejected_items[u] = bot

    return {
        "elig_mask": np.isfinite(chosen_score),
        "chosen_score": chosen_score,
        "rejected_score": rejected_score,
        "gap": chosen_score - rejected_score,
        "m_used": m_used,
        "chosen_items": chosen_items,
        "rejected_items": rejected_items,
    }


def stats_describe(x: np.ndarray, label: str) -> str:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return f"  {label:20s} empty"
    qs = np.quantile(x, [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    return (f"  {label:20s} n={x.size:>10,}  mean={x.mean():7.3f}  "
            f"min={qs[0]:6.2f}  p25={qs[1]:6.2f}  p50={qs[2]:6.2f}  "
            f"p75={qs[3]:6.2f}  p95={qs[5]:6.2f}  p99={qs[6]:6.2f}  max={qs[7]:6.2f}")


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def report_target_lengths(lengths: np.ndarray) -> None:
    section("1. Target length distribution & user eligibility")
    print(f"users (rows)                    : {len(lengths):,}")
    print(f"users with target len > 0       : {int((lengths > 0).sum()):,}")
    for L_min in [1, 4, 6, 8, 10]:
        n = int((lengths >= L_min).sum())
        print(f"users with target len >= {L_min:2d}     : {n:>10,}  ({n/len(lengths):.2%})")
    print()
    print("Length value counts:")
    vals, counts = np.unique(lengths, return_counts=True)
    for v, c in zip(vals, counts):
        print(f"  L={int(v):3d}: {int(c):>10,}  ({c/len(lengths):.2%})")
    print()
    print(f">>> Eligibility for pair construction: target len >= {2*MIN_HALF} (m_min={MIN_HALF}).")


def report_arm_pair_distributions(results: dict[str, dict]) -> None:
    section("2. Pair score distributions per arm")

    for name, r in results.items():
        print(f"\n--- {name} ---")
        cs = r["chosen_score"][r["elig_mask"]]
        rs = r["rejected_score"][r["elig_mask"]]
        gp = r["gap"][r["elig_mask"]]
        print(stats_describe(cs, "chosen_score"))
        print(stats_describe(rs, "rejected_score"))
        print(stats_describe(gp, "gap"))
        # Histogram of gap
        edges = np.array([0, 0.001, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, np.inf])
        idx = np.digitize(gp, edges) - 1
        print("  gap bucket distribution:")
        for k in range(len(edges) - 1):
            cnt = int((idx == k).sum())
            pct = cnt / max(gp.size, 1)
            bar = "#" * int(pct * 50)
            print(f"    [{edges[k]:5.2f}, {edges[k+1]:5.2f})  {cnt:>10,}  {pct:7.2%}  {bar}")


def report_filter_thresholds(results: dict[str, dict]) -> None:
    section("3. Surviving pair count under gap thresholds")
    print("How many users still have a usable pair if we require gap > T?\n")
    header = f"  {'arm':30s} " + "".join(f"{f'gap>{t}':>11s}" for t in GAP_THRESHOLDS)
    print(header)
    for name, r in results.items():
        gp = r["gap"][r["elig_mask"]]
        line = f"  {name:30s} "
        for t in GAP_THRESHOLDS:
            n = int((gp > t).sum())
            line += f"{n:>11,}"
        print(line)
    print()
    print("Same as % of eligible users:")
    print(header)
    for name, r in results.items():
        gp = r["gap"][r["elig_mask"]]
        denom = max(gp.size, 1)
        line = f"  {name:30s} "
        for t in GAP_THRESHOLDS:
            n = int((gp > t).sum())
            line += f"{n/denom:>10.2%} "
        print(line)


def report_cross_arm_consistency(results: dict[str, dict]) -> None:
    section("4. Cross-arm chosen-subset agreement")
    print("For each user (eligible under both arms), what fraction have")
    print("the EXACT SAME chosen item-set across two arms?")
    print("If two arms always pick the same chosen subset, the ablation has")
    print("no effect: the gap difference comes only from the score formula,")
    print("not from picking different items.\n")

    arm_names = list(results.keys())
    print("  " + " " * 32 + "  ".join(f"{n[:14]:>14s}" for n in arm_names))
    for a in arm_names:
        line = f"  {a:30s} "
        for b in arm_names:
            ra, rb = results[a], results[b]
            both = ra["elig_mask"] & rb["elig_mask"]
            if both.sum() == 0:
                line += f"{'-':>16s}"
                continue
            same = 0
            tot = 0
            for u in np.where(both)[0]:
                ca = ra["chosen_items"][u]
                cb = rb["chosen_items"][u]
                if ca is None or cb is None:
                    continue
                tot += 1
                if set(ca.tolist()) == set(cb.tolist()):
                    same += 1
            rate = same / tot if tot else 0
            line += f"  {rate:>14.2%}"
        print(line)


def report_arm_signal_usage(results: dict[str, dict], sig_full: np.ndarray,
                            offsets: np.ndarray) -> None:
    section("5. How often does each arm's negative penalty actually fire?")
    print("Arm 3 / Arm 4 carry a -2.0 not_interested term. We count how many")
    print("ELIGIBLE users have at least one not_interested item ANYWHERE in")
    print("their target sequence (whether it ends up in chosen or rejected).\n")

    ni_idx = SIGNALS.index("not_interested")
    elig_any = next(iter(results.values()))["elig_mask"]
    n_elig = int(elig_any.sum())
    has_ni = 0
    has_ni_in_rejected_arm3 = 0
    r3 = results["Arm3_full"]
    for u in np.where(elig_any)[0]:
        seg = sig_full[offsets[u]:offsets[u + 1], ni_idx]
        if seg.any():
            has_ni += 1
            rej = r3["rejected_items"][u]
            if rej is not None and seg[rej].any():
                has_ni_in_rejected_arm3 += 1
    print(f"  eligible users                              : {n_elig:>10,}")
    print(f"  ... with any not_interested in target       : {has_ni:>10,}  ({has_ni/n_elig:.4%})")
    print(f"  ... with not_interested ending up in REJECTED (Arm3): {has_ni_in_rejected_arm3:>10,}  ({has_ni_in_rejected_arm3/n_elig:.4%})")


def report_zero_gap(results: dict[str, dict]) -> None:
    section("6. Pairs with zero gap (degenerate)")
    print("If gap == 0, chosen and rejected have identical session score —")
    print("the pair carries no preference signal under that reward formula.\n")
    for name, r in results.items():
        gp = r["gap"][r["elig_mask"]]
        z = int((gp == 0).sum())
        print(f"  {name:30s} zero-gap users: {z:>10,}  ({z/max(gp.size,1):.2%})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    buf = StringIO()
    with redirect_stdout(buf):
        print("OpenOneRec — Script 3: DPO Feasibility Report")
        print(f"Generated by {os.path.basename(__file__)}")
        print(f"Data dir : {DATA_DIR}")

        df = pd.read_parquet(MASTER)
        df = df[df["split"] == 0].reset_index(drop=True)
        print(f"Loaded {len(df):,} split=0 rows")

        print("\nBuilding target signal matrix ...")
        sig, offsets, lengths = build_signal_matrix(df, "target_video")
        print(f"  total target items: {sig.shape[0]:,}")

        report_target_lengths(lengths)

        print("\nComputing pair scores under each arm ...")
        results = {}
        for name, weights in ARMS.items():
            results[name] = per_arm_chosen_rejected(sig, offsets, lengths, weights, MIN_HALF)
            n = int(results[name]["elig_mask"].sum())
            print(f"  {name:30s} eligible users: {n:,}")

        report_arm_pair_distributions(results)
        report_filter_thresholds(results)
        report_cross_arm_consistency(results)
        report_arm_signal_usage(results, sig, offsets)
        report_zero_gap(results)

        section("Summary")
        print("Pair-construction strategy:")
        print(f"  - target sub-session size m = L // 2")
        print(f"  - eligibility: target length >= {2*MIN_HALF}")
        print(f"  - chosen = top-m items by per-arm reward, rejected = bottom-m")
        print(f"  - same m across arms for direct comparability")
        print()
        print("Use the gap distributions and zero-gap rates to decide:")
        print("  (1) which arms have enough signal to train")
        print("  (2) what gap threshold to use for filtering")
        print("  (3) whether the four arms produce meaningfully different chosen sets")
        print()
        print("Done.")

    text = buf.getvalue()
    with open(OUT_PATH, "w") as f:
        f.write(text)
    sys.stdout.write(text)
    sys.stdout.write(f"\n[wrote report to {OUT_PATH}]\n")


if __name__ == "__main__":
    main()