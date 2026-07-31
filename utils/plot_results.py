"""utils/plot_results.py — Generador de figuras para el paper VSAT Monitor.

Produce (en paper/figures/):
  fig1_auroc_heatmap.pdf    — heatmap AUROC para cada señal x condición
  fig2_controls_bar.pdf     — AUROC de controles (shift / noise) por señal
  fig3_epsilon_trend.pdf    — AUROC vs epsilon con IC bootstrap
  fig4_signal_comparison.pdf— violin plot de distribución de AUROC por señal

Uso:
  python utils/plot_results.py \\
      --results outputs_qwen_overnight/results.json \\
      --out paper/figures

Requiere: matplotlib>=3.7, numpy
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Configuración global de estilo ──────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "legend.fontsize":    8.5,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "text.usetex":        False,
    "figure.dpi":         150,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

SIGNAL_LABELS = {
    "mahalanobis":  "Mahalanobis",
    "spectral":     "Spectral",
    "cosine":       "Cosine",
    "loss_shift":   "Loss Shift",
    "residual_pca": "Residual PCA",
}
ATTACK_LABELS = {
    "A1_lexical":   "A1 Lexical Trigger",
    "A2_labelflip": "A2 Label Flip",
    "A3_outfeat":   "A3 Output Feature",
}
EPSILONS = [0.01, 0.05, 0.1]
SIGNALS  = list(SIGNAL_LABELS.keys())
ATTACKS  = list(ATTACK_LABELS.keys())
# Order per figure, aligned with the paper tables:
# Fig 1 (heatmap) matches Table 4: Maha | RPCA | Spec | Cos | Loss
SIGNALS_HEATMAP  = ["mahalanobis", "residual_pca", "spectral", "cosine", "loss_shift"]
# Fig 2 (controls) matches Table 5: Maha | RPCA | Spec | Loss | Cosine
SIGNALS_CONTROLS = ["mahalanobis", "residual_pca", "spectral", "loss_shift", "cosine"]
PALETTE_SIGNALS = ["#E15759", "#76B7B2", "#EDC948", "#B07AA1", "#FF9DA7"]
PALETTE_ATTACKS = ["#4E79A7", "#F28E2B", "#59A14F"]


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Fig 1: Heatmap AUROC ────────────────────────────────────────────────────
def fig1_auroc_heatmap(data: dict, out_dir: Path):
    results = data["results"]
    row_labels, matrix, ci_lo, ci_hi = [], [], [], []
    for attack in ATTACKS:
        for eps in EPSILONS:
            key = f"{attack}@eps={eps}"
            if key not in results:
                continue
            row_labels.append(f"{ATTACK_LABELS[attack]}  ε={eps}")
            row, rlo, rhi = [], [], []
            for s in SIGNALS_HEATMAP:
                r = results[key].get(s, {})
                row.append(r.get("auroc", float("nan")))
                ci = r.get("auroc_ci", [float("nan"), float("nan")])
                rlo.append(ci[0])
                rhi.append(ci[1])
            matrix.append(row)
            ci_lo.append(rlo)
            ci_hi.append(rhi)

    mat = np.array(matrix)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    # Diverging map centered on chance (0.50): near-chance cells render
    # near-white, deviations tint blue (<0.5) or red (>0.5). RdBu is
    # colorblind-safe and stays monotonic-in-lightness from the center in
    # grayscale (B/N print). The figure's message — everything at chance —
    # reads as a uniform neutral field.
    im = ax.imshow(mat, vmin=0.35, vmax=0.65, cmap="RdBu_r", aspect="auto")

    ax.set_xticks(range(len(SIGNALS_HEATMAP)))
    ax.set_xticklabels([SIGNAL_LABELS[s] for s in SIGNALS_HEATMAP], rotation=22, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)

    # rejilla sutil entre celdas para estructura en B/N
    ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="0.85", linewidth=0.6)
    ax.tick_params(which="minor", length=0)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            col = "white" if abs(v - 0.5) > 0.10 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color=col)

    # separadores por ataque (cada 3 filas)
    for sep in [2.5, 5.5]:
        ax.axhline(sep, color="white", lw=2.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("AUROC")
    cbar.set_ticks([0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65])
    ax.set_title(
        "AUROC per attack condition × detection signal\n"
        "Model: Qwen2.5-1.5B · Dataset: HH-RLHF · Adapter: LoRA r=8 · "
        "B=64 · 3 seeds × 12 eval batches",
        fontsize=9, pad=6
    )
    fig.tight_layout()
    p = out_dir / "fig1_auroc_heatmap.pdf"
    fig.savefig(p)
    plt.close(fig)
    print(f"  ✓ {p}")


# ── Fig 2: Control bars ─────────────────────────────────────────────────────
def fig2_controls_bar(data: dict, out_dir: Path):
    controls = data["controls"]
    signals_in = [s for s in SIGNALS_CONTROLS if s in controls]
    shift = [controls[s]["shift_vs_clean_auroc"] for s in signals_in]
    noise = [controls[s]["noise_vs_clean_auroc"] for s in signals_in]

    x = np.arange(len(signals_in))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.bar(x - width/2, shift, width, label="Domain shift (benign)",
           color="#4E79A7", alpha=0.85, edgecolor="white")
    ax.bar(x + width/2, noise, width, label="Label noise 10% (benign)",
           color="#F28E2B", alpha=0.85, edgecolor="white")

    ax.axhline(0.5, color="grey", linestyle="--", lw=1.2, label="Random baseline (0.50)")
    ax.axhline(0.10, color="#76B7B2", linestyle=":", lw=1.2, label="Target ≤ 0.10")

    ax.set_xticks(x)
    ax.set_xticklabels([SIGNAL_LABELS[s] for s in signals_in], rotation=18, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("AUROC vs. clean batches")
    ax.set_title("False-alarm control: detection of benign distribution shifts\n"
                 "(ideal detector → 0.00 for shift, ≈ 0.50 for i.i.d. noise)")
    ax.legend(fontsize=8, loc="upper left")

    # annotate cosine failure
    if "cosine" in signals_in:
        ci = signals_in.index("cosine")
        ax.annotate("Cosine\nfalse-alarm\nAUROC = 1.0",
                    xy=(ci - width/2, 1.0), xytext=(ci - width/2 - 1.7, 0.8),
                    arrowprops=dict(arrowstyle="->", color="red"),
                    color="red", fontsize=8)
    fig.tight_layout()
    p = out_dir / "fig2_controls_bar.pdf"
    fig.savefig(p)
    plt.close(fig)
    print(f"  ✓ {p}")


# ── Fig 3: AUROC vs epsilon trend ───────────────────────────────────────────
def fig3_epsilon_trend(data: dict, out_dir: Path):
    results = data["results"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)

    for ax, attack in zip(axes, ATTACKS):
        for sig, sc in zip(SIGNALS, PALETTE_SIGNALS):
            aurocs, cis_lo, cis_hi = [], [], []
            for eps in EPSILONS:
                key = f"{attack}@eps={eps}"
                if key not in results or sig not in results[key]:
                    aurocs.append(np.nan); cis_lo.append(np.nan); cis_hi.append(np.nan)
                    continue
                r = results[key][sig]
                aurocs.append(r["auroc"])
                cis_lo.append(r["auroc_ci"][0])
                cis_hi.append(r["auroc_ci"][1])

            eps_arr = np.array(EPSILONS)
            aurocs = np.array(aurocs)
            ax.plot(eps_arr, aurocs, marker="o", color=sc, linewidth=1.8,
                    label=SIGNAL_LABELS[sig], markersize=5)
            ax.fill_between(eps_arr, cis_lo, cis_hi, alpha=0.13, color=sc)

        ax.axhline(0.5, color="grey", linestyle="--", lw=1.0)
        ax.set_title(ATTACK_LABELS[attack], fontsize=9.5)
        ax.set_xlabel("Poison rate ε", fontsize=9)
        ax.set_ylim(0.28, 0.78)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("AUROC")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle(
        "AUROC vs. poison rate ε per attack and detection signal "
        "(shaded = 95% bootstrap CI, 3 seeds)",
        fontsize=9.5, y=1.02
    )
    fig.tight_layout()
    p = out_dir / "fig3_epsilon_trend.pdf"
    fig.savefig(p)
    plt.close(fig)
    print(f"  ✓ {p}")


# ── Fig 4: Violin signal comparison ─────────────────────────────────────────
def fig4_signal_comparison(data: dict, out_dir: Path):
    results = data["results"]
    signal_vals = {s: [] for s in SIGNALS}
    for key, row in results.items():
        for s in SIGNALS:
            if s in row:
                signal_vals[s].append(row[s]["auroc"])

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    positions = np.arange(len(SIGNALS))
    vals = [signal_vals[s] for s in SIGNALS]
    parts = ax.violinplot(vals, positions=positions, showmedians=True, showextrema=True)

    for pc, color in zip(parts["bodies"], PALETTE_SIGNALS):
        pc.set_facecolor(color)
        pc.set_alpha(0.72)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)

    ax.axhline(0.5, color="grey", linestyle="--", lw=1.2, label="Random baseline")
    ax.set_xticks(positions)
    ax.set_xticklabels([SIGNAL_LABELS[s] for s in SIGNALS])
    ax.set_ylabel("AUROC (all 9 attack conditions)")
    ax.set_title("Distribution of AUROC across all attack conditions per detection signal\n"
                 "(Qwen2.5-1.5B · LoRA r=8 · HH-RLHF)")
    ax.legend()
    ax.set_ylim(0.28, 0.78)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    p = out_dir / "fig4_signal_comparison.pdf"
    fig.savefig(p)
    plt.close(fig)
    print(f"  ✓ {p}")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Generate VSAT paper figures from results.json.")
    ap.add_argument("--results", default="outputs_qwen_overnight/results.json")
    ap.add_argument("--out", default="paper/figures")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.results}...")
    data = load(args.results)
    print("Generating figures:")
    fig1_auroc_heatmap(data, out_dir)
    fig2_controls_bar(data, out_dir)
    fig3_epsilon_trend(data, out_dir)
    fig4_signal_comparison(data, out_dir)
    print(f"\nAll figures written to {out_dir}/")


if __name__ == "__main__":
    main()
