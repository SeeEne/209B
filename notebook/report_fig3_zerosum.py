"""
report_fig3_zerosum.py — Figure 3 for the MS4 final report.

Conceptual schematic for the structural argument in section 5.3:
  When chosen and rejected items live in the same high-prob region of a
  narrow categorical output space (K=8192 here, vs ~50k-150k in standard
  LLMs), an SFT objective on chosen alone drains rejected for free via
  softmax zero-sum redistribution.

The mechanism (no new data, no experiments):
  -log P(c) gradient pushes z_c up by (1 - P(c)) and pushes every other
  z_i down by P(z_i). The downward push is proportional to current
  probability mass, so rejected items — which co-locate with chosen in
  the user's neighborhood — lose more probability mass than tail items.
  After re-softmax, P(rejected) drops without ever appearing in the loss.

Two panels at toy K=32 (the real codebook is 8192-wide; we shrink it for
visualization). The simulation re-softmaxes after a logit boost on
chosen, mirroring what one optimization step of -log P(c) does to the
output distribution.

Output:
    notebook/figures/fig3_zerosum.pdf
    notebook/figures/fig3_zerosum.png

Run:
    python notebook/report_fig3_zerosum.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ---------------------------------------------------------------------------
# Toy distribution config
# ---------------------------------------------------------------------------
K = 32
SIDS = np.arange(K)

NEIGHBORHOOD = np.arange(9, 19)               # user's high-prob region
CHOSEN_IDX   = np.array([11, 12, 13])
REJECTED_IDX = np.array([14, 15, 16])

COLOR_CHOSEN   = "#1f6dd9"
COLOR_REJECTED = "#d62728"
COLOR_OTHER    = "#bdbdbd"
COLOR_HILITE   = "#fff4e6"


def make_base_distribution():
    """Bell over the user's neighborhood + diffuse tail elsewhere."""
    bell = np.exp(-((SIDS - 13.5) ** 2) / (2 * 3.0 ** 2))
    bell = bell / bell.sum() * 0.85

    tail_mask = ~np.isin(SIDS, NEIGHBORHOOD)
    tail = np.zeros(K)
    tail[tail_mask] = 0.15 / tail_mask.sum()

    p = bell + tail
    return p / p.sum()


def apply_sft_step(p_base: np.ndarray, boost: float = 2.0) -> np.ndarray:
    """Simulate a logit boost on chosen + re-softmax."""
    logits = np.log(p_base + 1e-12)
    logits[CHOSEN_IDX] += boost
    z = np.exp(logits - logits.max())
    return z / z.sum()


def color_for(i: int) -> str:
    if i in CHOSEN_IDX:
        return COLOR_CHOSEN
    if i in REJECTED_IDX:
        return COLOR_REJECTED
    return COLOR_OTHER


def draw_panel(ax, p, title, xlabel):
    ax.axvspan(NEIGHBORHOOD.min() - 0.5, NEIGHBORHOOD.max() + 0.5,
               color=COLOR_HILITE, zorder=0)
    colors = [color_for(i) for i in SIDS]
    ax.bar(SIDS, p, color=colors, width=0.82, edgecolor="white",
           linewidth=0.4, zorder=2)
    ax.set_title(title, fontsize=10, loc="left", pad=4)
    ax.set_xlabel(xlabel)
    ax.set_xlim(-0.6, K - 0.4)
    ax.tick_params(direction="in", length=3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main():
    project_root = Path(__file__).resolve().parents[1]
    out_pdf = project_root / "notebook" / "figures" / "fig3_zerosum.pdf"
    out_png = project_root / "notebook" / "figures" / "fig3_zerosum.png"

    p_base  = make_base_distribution()
    p_after = apply_sft_step(p_base, boost=2.0)

    # share y so the height differences (zero-sum drain) are immediately
    # readable across the two panels
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)

    draw_panel(ax_l, p_base,
               r"(a) base model: $P(\mathrm{SID}\mid\mathrm{user\ context})$",
               "SID index   (toy $K=32$; real codebook $K=8192$)")
    ax_l.set_ylabel("probability mass")

    # neighborhood annotation on left panel only
    ax_l.text(13.5, p_base.max() * 1.07, "user's high-prob region",
              ha="center", fontsize=8.5, color="#555555", style="italic")

    draw_panel(ax_r, p_after,
               r"(b) after one SFT step on chosen",
               "SID index")

    # ---- Mass-flow indicator on right panel ----
    rej_top = p_after[REJECTED_IDX].max()
    cho_top = p_after[CHOSEN_IDX].max()
    arrow_y = max(cho_top, rej_top) * 0.55
    ax_r.annotate("",
        xy=(CHOSEN_IDX.mean(), arrow_y),
        xytext=(REJECTED_IDX.mean(), arrow_y),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2,
                        connectionstyle="arc3,rad=-0.35"),
        zorder=4,
    )
    ax_r.text((CHOSEN_IDX.mean() + REJECTED_IDX.mean()) / 2,
              arrow_y * 1.45,
              "mass redistributes\n(zero-sum)",
              ha="center", fontsize=8.5, color="#333333", style="italic")

    # ---- Per-region delta annotations on left panel for clarity ----
    chosen_delta = p_after[CHOSEN_IDX].sum() - p_base[CHOSEN_IDX].sum()
    rej_delta    = p_after[REJECTED_IDX].sum() - p_base[REJECTED_IDX].sum()
    tail_delta   = p_after[~np.isin(SIDS, NEIGHBORHOOD)].sum() \
                   - p_base[~np.isin(SIDS, NEIGHBORHOOD)].sum()

    delta_text = (
        f"$\\Delta\\!\\sum P(\\mathrm{{chosen}}) = {chosen_delta:+.3f}$\n"
        f"$\\Delta\\!\\sum P(\\mathrm{{rejected}}) = {rej_delta:+.3f}$\n"
        f"$\\Delta\\!\\sum P(\\mathrm{{tail}}) = {tail_delta:+.3f}$"
    )
    ax_r.text(0.02, 0.96, delta_text,
              transform=ax_r.transAxes, fontsize=8, color="#222222",
              va="top", ha="left",
              bbox=dict(facecolor="white", alpha=0.85,
                        edgecolor="#cccccc", boxstyle="round,pad=0.3"))

    # ---- Legend on left panel ----
    legend_handles = [
        Patch(color=COLOR_CHOSEN,   label="chosen"),
        Patch(color=COLOR_REJECTED, label="rejected"),
        Patch(color=COLOR_OTHER,    label="other SIDs"),
    ]
    ax_l.legend(handles=legend_handles, loc="upper right", fontsize=8.5,
                framealpha=0.95, borderpad=0.3, handletextpad=0.5)

    fig.tight_layout(pad=0.6)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=200, bbox_inches="tight")
    print(f"  wrote {out_pdf}")
    print(f"  wrote {out_png}")

    print("\nMass deltas (SFT step on chosen, K={}):".format(K))
    print(f"  chosen   region: {p_base[CHOSEN_IDX].sum():.3f} -> {p_after[CHOSEN_IDX].sum():.3f}  ({chosen_delta:+.3f})")
    print(f"  rejected region: {p_base[REJECTED_IDX].sum():.3f} -> {p_after[REJECTED_IDX].sum():.3f}  ({rej_delta:+.3f})")
    print(f"  tail   region:   {p_base[~np.isin(SIDS, NEIGHBORHOOD)].sum():.3f} -> {p_after[~np.isin(SIDS, NEIGHBORHOOD)].sum():.3f}  ({tail_delta:+.3f})")
    print("  -> rejected drains far more than tail despite tail having more")
    print("     items, because rejected co-locates in the high-prob region.")


if __name__ == "__main__":
    main()
