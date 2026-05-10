"""
report_fig2_scatter.py — Figure 2 for the MS4 final report.

Plots recall_chosen vs recall_rejected for the four arms evaluated on the
full v1 valid set (n=7940). The diagonal y=x partitions the plane into
"anti-aligned" (above) and "aligned" (below). Reading the figure:

    above diagonal  : recall_rejected > recall_chosen (model surfaces non-
                      engaged items more than engaged ones — anti-aligned).
                      OneRec baseline lives here.
    below diagonal  : recall_chosen > recall_rejected (model prefers engaged).
                      Only DPO+SFT-anchor enters this region — but at the
                      cost of crashing both chosen and rejected absolute
                      recall (lower-left corner).
    on diagonal     : neutral — SFT-50k and ORPO 50k tie here, having
                      debiased the baseline's anti-alignment but not yet
                      pushed into positive preference.

Inputs:
    evaluation_results/eval_engaged_baseline_full.json
    evaluation_results/eval_engaged_sft50k_full.json
    evaluation_results/eval_engaged_orpo_full.json
    evaluation_results/eval_engaged_dpo_anchor_full.json

Output:
    notebook/figures/fig2_scatter.pdf
    notebook/figures/fig2_scatter.png  (for quick preview)

Run:
    python notebook/report_fig2_scatter.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Config — keep colors / style consistent with figure 1.
# ---------------------------------------------------------------------------

ARMS = [
    # (label,            json_basename,                     color,    marker, label_offset (x_off, y_off, ha, va))
    # SFT and ORPO points sit ~0.0002 apart on both axes so we offset their
    # labels in opposite quadrants to keep them readable.
    ("OneRec baseline",  "eval_engaged_baseline_full.json", "#666666", "o",   ( 0.0010,  0.0000, "left",  "center")),
    ("SFT-50k",          "eval_engaged_sft50k_full.json",   "#1f6dd9", "s",   ( 0.0010,  0.0010, "left",  "bottom")),
    ("ORPO 50k",         "eval_engaged_orpo_full.json",     "#2ca02c", "^",   (-0.0010, -0.0012, "right", "top")),
    ("DPO + SFT-anchor", "eval_engaged_dpo_anchor_full.json", "#d62728", "D", ( 0.0010,  0.0000, "left",  "center")),
]


def load_arms(eval_dir: Path):
    """Read each arm's summary JSON, return list of dicts with x/y/etc."""
    points = []
    for label, fname, color, marker, offset in ARMS:
        fp = eval_dir / fname
        if not fp.exists():
            print(f"  [skip] {fp} missing")
            continue
        d = json.loads(fp.read_text())
        points.append({
            "label": label,
            "x": d["recall_chosen"],
            "y": d["recall_rejected"],
            "color": color,
            "marker": marker,
            "offset": offset,
            "n": d["n"],
        })
    return points


def plot(points, out_pdf: Path, out_png: Path):
    # Square axes — NeurIPS half-page-width-ish (3.6") for Results section.
    fig, ax = plt.subplots(figsize=(3.7, 3.6))

    # ---- Diagonal reference y = x ----
    # Range chosen so all 4 points + a small margin are visible.
    lim_lo, lim_hi = 0.000, 0.030
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
            color="#bbbbbb", linestyle="--", linewidth=0.9, zorder=1,
            label=r"$y = x$ (no discrimination)")

    # Light shading above-diagonal = anti-aligned region (where baseline lives)
    ax.fill_between([lim_lo, lim_hi], [lim_lo, lim_hi], [lim_hi, lim_hi],
                    color="#cccccc", alpha=0.18, zorder=0)
    # Keep region labels in the upper-left / lower-right corners well clear
    # of any plotted point (SFT/ORPO at ~0.014, baseline at 0.025-rejected,
    # DPO+anchor near origin).
    ax.text(0.0010, 0.0285, "anti-aligned\n(rejected > chosen)",
            color="#666666", fontsize=8, ha="left", va="top", style="italic")
    ax.text(0.0290, 0.0015, "aligned\n(chosen > rejected)",
            color="#666666", fontsize=8, ha="right", va="bottom", style="italic")

    # ---- Per-arm points ----
    for p in points:
        ax.scatter(p["x"], p["y"], s=70, c=p["color"], marker=p["marker"],
                   edgecolors="white", linewidths=1.0, zorder=3,
                   label=f"{p['label']} (n={p['n']})")
        dx, dy, ha, va = p["offset"]
        ax.annotate(p["label"], (p["x"], p["y"]),
                    xytext=(p["x"] + dx, p["y"] + dy),
                    fontsize=9, ha=ha, va=va, zorder=4,
                    color="#222222")

    # ---- Axes ----
    ax.set_xlabel(r"recall$_{\mathrm{chosen}}$ @ top-K=96")
    ax.set_ylabel(r"recall$_{\mathrm{rejected}}$ @ top-K=96")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(direction="in", length=3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Tight, paper-quality save.
    fig.tight_layout(pad=0.4)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=200, bbox_inches="tight")
    print(f"  wrote {out_pdf}")
    print(f"  wrote {out_png}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    eval_dir = project_root / "evaluation_results"
    out_pdf = project_root / "notebook" / "figures" / "fig2_scatter.pdf"
    out_png = project_root / "notebook" / "figures" / "fig2_scatter.png"

    print(f"reading from {eval_dir}")
    points = load_arms(eval_dir)
    if len(points) < 4:
        print(f"  warning: only {len(points)}/4 arms found — figure may be incomplete")

    plot(points, out_pdf, out_png)
    print()
    print("Coordinates of plotted points (for caption / report):")
    for p in points:
        delta = p["x"] - p["y"]
        loc = "above diagonal (anti-aligned)" if delta < 0 else "below or on diagonal"
        print(f"  {p['label']:<22}  chosen={p['x']:.4f}  rejected={p['y']:.4f}  "
              f"Δ={delta:+.4f}  ({loc})")


if __name__ == "__main__":
    main()
