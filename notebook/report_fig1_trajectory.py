"""
report_fig1_trajectory.py — Figure 1 for the MS4 final report.

Two-panel layout:
  (a) eval log P(chosen) per token  — three 50k arms
  (b) eval log P(rejected) per token — ORPO 50k vs DPO + SFT-anchor 50k
      (SFT-only is omitted because the SFT trainer does not log
       eval_rejected_score; SFT's effect on rejected is read off
       Table 1's r_r column instead.)

Step-0 anchors:
    SFT-50k   chosen   ≈ -4.920  (OneRec base)
    ORPO 50k  chosen   ≈ -4.920  (OneRec base)
    DPO+anchor chosen  ≈ -4.795  (Stage 2 starts from SFT-50k endpoint)
    ORPO 50k  rejected ≈ -4.920  (base hasn't differentiated chosen vs rejected)
    DPO+anchor rejected ≈ -4.920 (SFT-50k didn't supervise rejected, so
                                  SFT-50k's endpoint rejected ≈ base rejected)

Visual stories:
  (a) SFT-50k and ORPO 50k climb together monotonically; DPO+SFT-anchor
      regresses ~1.5 nat below where it started despite the 0.15 anchor.
  (b) ORPO's OR term, despite its design, has no measurable effect on
      rejected log P (flat at -4.99 across all 5 ckpts); DPO+anchor's DPO
      term drives rejected ~5.5 nat below base. The drop in rejected
      recall reported for ORPO in Table 1 (0.0246 -> 0.0143) cannot be
      attributed to OR-term gradient since OR-term gradient is ~0 — it is
      attributable to NLL's zero-sum redistribution (see sec 5.3).

Inputs (per-arm checkpoint trainer_state.json):
    runs/sft_only_50k/checkpoint-6250/trainer_state.json
    runs/orpo_50k/checkpoint-6250/trainer_state.json
    runs/dpo_anchor_from_sft_50k/checkpoint-6250/trainer_state.json

Output:
    notebook/figures/fig1_trajectory.pdf
    notebook/figures/fig1_trajectory.png

Run:
    python notebook/report_fig1_trajectory.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Step-0 anchors (see file docstring for derivation).
# ---------------------------------------------------------------------------
BASE_CHOSEN_SCORE   = -4.920
BASE_REJECTED_SCORE = -4.920
SFT_50K_CHOSEN_END  = -4.795
# SFT-only didn't supervise rejected; its endpoint rejected logP is
# approximately unchanged from base.
SFT_50K_REJECTED_END = -4.920

COLOR_SFT  = "#1f6dd9"
COLOR_ORPO = "#2ca02c"
COLOR_DPO  = "#d62728"

# (label, run_dir, color, marker, step_0_value)
ARMS_CHOSEN = [
    ("SFT-50k",          "runs/sft_only_50k",            COLOR_SFT,  "s", BASE_CHOSEN_SCORE),
    ("ORPO 50k",         "runs/orpo_50k",                COLOR_ORPO, "^", BASE_CHOSEN_SCORE),
    ("DPO + SFT-anchor", "runs/dpo_anchor_from_sft_50k", COLOR_DPO,  "D", SFT_50K_CHOSEN_END),
]

# SFT-only is intentionally absent: its trainer doesn't log eval_rejected_score.
ARMS_REJECTED = [
    ("ORPO 50k",         "runs/orpo_50k",                COLOR_ORPO, "^", BASE_REJECTED_SCORE),
    ("DPO + SFT-anchor", "runs/dpo_anchor_from_sft_50k", COLOR_DPO,  "D", SFT_50K_REJECTED_END),
]


def load_trajectory(run_dir: Path, key: str, step_0_value: float):
    ckpt_dirs = list(run_dir.glob("checkpoint-*"))
    if not ckpt_dirs:
        return None
    final_ckpt = max(ckpt_dirs, key=lambda p: int(p.name.split("-")[1]))
    state = json.loads((final_ckpt / "trainer_state.json").read_text())
    eval_entries = [(e["step"], e[key]) for e in state["log_history"] if key in e]
    return [(0, step_0_value)] + eval_entries


def style_axes(ax, base_value, base_label_x):
    ax.axhline(base_value, color="#999999", linestyle=":", linewidth=0.9, zorder=1)
    ax.text(base_label_x, base_value, "  base",
            color="#666666", fontsize=8, va="center")
    ax.set_xlabel("Training step")
    ax.set_xlim(-200, 7100)
    ax.set_xticks([0, 1250, 2500, 3750, 5000, 6250])
    ax.tick_params(direction="in", length=3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_trajectories(ax, arm_data):
    for entry in arm_data:
        traj = entry["traj"]
        xs = [s for s, _ in traj]
        ys = [v for _, v in traj]
        ax.plot(xs, ys, color=entry["color"], marker=entry["marker"],
                markersize=5, linewidth=1.5, markeredgecolor="white",
                markeredgewidth=0.6, label=entry["label"], zorder=3)


def annotate_dpo_regression(ax, arm_data):
    dpo = next((a for a in arm_data if "DPO" in a["label"]), None)
    if not dpo or len(dpo["traj"]) <= 1:
        return
    start = dpo["traj"][0]
    worst = min(dpo["traj"][1:], key=lambda p: p[1])
    drop = start[1] - worst[1]
    # Always place text above the worst point with a fixed offset; the
    # offset is in nats so it scales with each panel's y-range naturally.
    text_y = worst[1] + 0.70
    ax.annotate(
        f"{drop:.2f} nat below start",
        xy=worst, xytext=(3300, text_y),
        fontsize=8, color=COLOR_DPO, ha="left",
        arrowprops=dict(arrowstyle="->", color=COLOR_DPO,
                        lw=0.8, shrinkA=0, shrinkB=4),
    )


def annotate_orpo_flat(ax, arm_data):
    orpo = next((a for a in arm_data if "ORPO" in a["label"]), None)
    if not orpo or len(orpo["traj"]) <= 1:
        return
    last = orpo["traj"][-1]
    ax.annotate(
        "OR term: no\nmeasurable\neffect",
        xy=last, xytext=(3700, -7.2),
        fontsize=8, color=COLOR_ORPO, ha="left",
        arrowprops=dict(arrowstyle="->", color=COLOR_ORPO,
                        lw=0.8, shrinkA=0, shrinkB=4),
    )


def main():
    project_root = Path(__file__).resolve().parents[1]
    out_pdf = project_root / "notebook" / "figures" / "fig1_trajectory.pdf"
    out_png = project_root / "notebook" / "figures" / "fig1_trajectory.png"

    chosen_data = []
    for label, run_dir, color, marker, step_0 in ARMS_CHOSEN:
        traj = load_trajectory(project_root / run_dir, "eval_chosen_score", step_0)
        if traj is None:
            print(f"  [skip] {run_dir} has no checkpoints")
            continue
        chosen_data.append({"label": label, "color": color, "marker": marker, "traj": traj})

    rejected_data = []
    for label, run_dir, color, marker, step_0 in ARMS_REJECTED:
        traj = load_trajectory(project_root / run_dir, "eval_rejected_score", step_0)
        if traj is None:
            print(f"  [skip] {run_dir} has no checkpoints")
            continue
        rejected_data.append({"label": label, "color": color, "marker": marker, "traj": traj})

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.5, 3.3))

    # ----- Left panel: chosen log P -----
    plot_trajectories(ax_l, chosen_data)
    annotate_dpo_regression(ax_l, chosen_data)
    style_axes(ax_l, BASE_CHOSEN_SCORE, base_label_x=6500)
    ax_l.set_ylabel(r"eval $\log P_\theta(\mathrm{chosen})$ per token  (nats)")
    ax_l.set_title("(a) chosen log P", fontsize=10, loc="left", pad=4)
    ax_l.legend(loc="lower left", fontsize=8.5, framealpha=0.95,
                borderpad=0.3, handletextpad=0.4)

    # ----- Right panel: rejected log P -----
    plot_trajectories(ax_r, rejected_data)
    annotate_dpo_regression(ax_r, rejected_data)
    annotate_orpo_flat(ax_r, rejected_data)
    style_axes(ax_r, BASE_REJECTED_SCORE, base_label_x=6500)
    ax_r.set_ylabel(r"eval $\log P_\theta(\mathrm{rejected})$ per token  (nats)")
    ax_r.set_title("(b) rejected log P", fontsize=10, loc="left", pad=4)
    ax_r.legend(loc="lower left", fontsize=8.5, framealpha=0.95,
                borderpad=0.3, handletextpad=0.4)

    fig.tight_layout(pad=0.6)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=200, bbox_inches="tight")
    print(f"  wrote {out_pdf}")
    print(f"  wrote {out_png}")

    print("\nLeft panel — chosen log P trajectories:")
    for a in chosen_data:
        traj_str = "  ".join(f"{s}:{v:+.3f}" for s, v in a["traj"])
        print(f"  {a['label']:<22} {traj_str}")
    print("\nRight panel — rejected log P trajectories:")
    for a in rejected_data:
        traj_str = "  ".join(f"{s}:{v:+.3f}" for s, v in a["traj"])
        print(f"  {a['label']:<22} {traj_str}")


if __name__ == "__main__":
    main()
