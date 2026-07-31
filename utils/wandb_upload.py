"""utils/wandb_upload.py — Sube los resultados de VSAT Monitor a Weights & Biases.

Crea un run con:
  - config completa del experimento
  - AUROC por condición (ataque × epsilon) como métricas individuales
  - AUROC de controles
  - tabla interactiva de resultados completos
  - artefactos: results.json, config.json, y figuras si existen

Uso:
  pip install wandb
  wandb login
  python utils/wandb_upload.py \\
      --results outputs_qwen_overnight/results.json \\
      --project vsat-monitor \\
      --name qwen25-1.5B-lora-r8-B64

  # Subir también figuras si ya fueron generadas:
  python utils/wandb_upload.py \\
      --results outputs_qwen_overnight/results.json \\
      --figures paper/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def upload(results_path: str, project: str, name: str, figures_dir: str | None):
    try:
        import wandb
    except ImportError:
        raise SystemExit("wandb not installed. Run: pip install wandb")

    data = load(results_path)
    cfg  = data["config"]

    run = wandb.init(
        project=project,
        name=name,
        config=cfg,
        tags=["vsat", "lora-dpo", "poison-detection", cfg["model"].replace("/", "-")],
    )

    # ── Métricas por condición ───────────────────────────────────────────────
    for key, row in data["results"].items():
        for signal, metrics in row.items():
            prefix = f"{key}/{signal}"
            run.log({
                f"{prefix}/auroc":        metrics["auroc"],
                f"{prefix}/auroc_ci_lo":  metrics["auroc_ci"][0],
                f"{prefix}/auroc_ci_hi":  metrics["auroc_ci"][1],
                f"{prefix}/det_fpr0.01":  metrics.get("det@fpr0.01", 0),
                f"{prefix}/det_fpr0.05":  metrics.get("det@fpr0.05", 0),
            })

    # ── Métricas de controles ────────────────────────────────────────────────
    for signal, ctrl in data["controls"].items():
        run.log({
            f"controls/{signal}/shift_vs_clean_auroc": ctrl["shift_vs_clean_auroc"],
            f"controls/{signal}/noise_vs_clean_auroc": ctrl["noise_vs_clean_auroc"],
        })

    # ── Tabla interactiva resumen ─────────────────────────────────────────────
    SIGNALS = ["mahalanobis", "spectral", "cosine", "loss_shift", "residual_pca"]
    table_data = []
    for key, row in data["results"].items():
        parts = key.split("@")
        attack = parts[0]
        eps    = float(parts[1].replace("eps=", ""))
        table_row = [attack, eps]
        for sig in SIGNALS:
            table_row.append(row.get(sig, {}).get("auroc", float("nan")))
        table_data.append(table_row)

    cols = ["attack", "epsilon"] + [f"auroc_{s}" for s in SIGNALS]
    table = wandb.Table(columns=cols, data=table_data)
    run.log({"auroc_summary_table": table})

    # ── Artefactos JSON ───────────────────────────────────────────────────────
    artifact = wandb.Artifact(
        name=f"results-{name}",
        type="experiment-results",
        description="VSAT Monitor Fase 1 — AUROC results and config",
    )
    artifact.add_file(results_path, name="results.json")
    out_dir = Path(results_path).parent
    config_path = out_dir / "config.json"
    if config_path.exists():
        artifact.add_file(str(config_path), name="config.json")
    run.log_artifact(artifact)

    # ── Figuras (si existen) ─────────────────────────────────────────────────
    if figures_dir:
        fig_path = Path(figures_dir)
        for pdf in sorted(fig_path.glob("*.pdf")):
            run.log({pdf.stem: wandb.Image(str(pdf))})
            print(f"  ✓ Uploaded figure: {pdf.name}")

    run.finish()
    print(f"\n✅ Run '{name}' uploaded to project '{project}' on W&B.")
    print(f"   URL: {run.url}")


def main():
    ap = argparse.ArgumentParser(description="Upload VSAT results to Weights & Biases.")
    ap.add_argument("--results",  default="outputs_qwen_overnight/results.json")
    ap.add_argument("--project",  default="vsat-monitor")
    ap.add_argument("--name",     default="qwen25-1.5B-lora-r8-B64")
    ap.add_argument("--figures",  default=None,
                    help="Directory with generated PDF figures to upload.")
    args = ap.parse_args()
    upload(args.results, args.project, args.name, args.figures)


if __name__ == "__main__":
    main()
