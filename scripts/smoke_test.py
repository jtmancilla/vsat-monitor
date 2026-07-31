"""Smoke test end-to-end con tiny-gpt2 y datos sinteticos (offline, CPU).

Verifica el pipeline completo: datos -> ataques -> DPO -> probe -> perfil ->
senales -> metricas. No pretende AUROC altos; pretende que todo corra.

Uso:  python scripts/smoke_test.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from vsat.config import Config
from vsat.data import load_pairs
from vsat.dpo import train_dpo
from vsat.experiment import run_experiment
from vsat.models import build_model_and_tokenizer


def main():
    cfg = Config(
        n_clean_train=160, n_profile_batches=20, batch_size=16,
        dpo_steps=40, proj_dim=128,
        epsilons=(0.05, 0.10), seeds=(0, 1),   # barrido reducido
        out_dir=os.path.join(os.path.dirname(__file__), "..", "outputs"),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    pairs = load_pairs(cfg)
    texts = []
    for p in pairs:
        texts += [p.prompt, p.chosen, p.rejected]
    texts += [cfg.trigger_token, cfg.trait_phrase]  # que el veneno sea tokenizable

    print("[setup] construyendo modelo y tokenizer ...")
    model, ref_model, tok = build_model_and_tokenizer(cfg, corpus_texts=texts)
    model.to(device)
    ref_model.to(device)
    print(f"  vocab={len(tok)}  params={sum(p.numel() for p in model.parameters())}")

    print("[setup] entrenando checkpoint DPO limpio ...")
    t0 = time.time()
    model = train_dpo(model, ref_model, pairs[:cfg.n_clean_train], tok, cfg, device)
    print(f"  ({time.time()-t0:.1f}s)")

    print("[experimento] barrido ...")
    t0 = time.time()
    out = run_experiment(model, ref_model, tok, pairs, cfg, device,
                         n_eval_batches=4)
    print(f"  ({time.time()-t0:.1f}s)")

    # Tabla resumen
    print("\n=== AUROC por ataque y senal ===")
    for key, row in out["results"].items():
        line = f"{key:28s}"
        for s in ("mahalanobis", "spectral", "cosine", "loss_shift"):
            line += f"  {s[:6]}={row[s]['auroc']:.2f}"
        print(line)
    print("\n=== Controles (AUROC control-vs-limpio; ~0.5 = sin falsas alarmas) ===")
    for s, c in out["controls"].items():
        print(f"{s:12s} shift={c['shift_vs_clean_auroc']:.2f} "
              f"noise={c['noise_vs_clean_auroc']:.2f}")


if __name__ == "__main__":
    main()
