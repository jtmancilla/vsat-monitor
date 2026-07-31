"""Medicion de overhead del VSAT Monitor (RQ4) — sin re-entrenar nada.

Carga el checkpoint DPO ya entrenado (run_dir/dpo_checkpoint) y mide, en el
mismo hardware y con la misma configuracion del run original (run_dir/config.json):

  1. Costo del shadow probe por par (forward policy+ref, backward sobre
     parametros monitoreados, proyeccion JL) — fases 2-4 del paper.
  2. Costo de un paso real de entrenamiento DPO por par (mismo loop interno
     que vsat.dpo.train_dpo: micro-batching + AdamW).
  3. Costo del scoring por lote (5 senales sobre Z ya proyectado).
  4. Escalado del ajuste Ledoit-Wolf con N_c (matrices aleatorias de las
     mismas dimensiones; el costo de LW es data-independent).

El ratio overhead = t_probe_por_par / t_train_por_par es el "x training time"
del paper: monitor y entrenamiento procesan cada par exactamente una vez.

Uso:
  python scripts/time_monitor.py --run-dir outputs_qwen_overnight \
      --reps 5 --train-steps 5

Salida: run_dir/timing_rq4.json + resumen por consola con frase para el paper.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from vsat.config import Config
from vsat.data import load_pairs
from vsat.dpo import dpo_losses
from vsat.models import build_model_and_tokenizer, monitored_params
from vsat.probe import probe_batch, make_projection
from vsat.profile import GradientProfile
from vsat.experiment import _batch_scores
from vsat.signals import LossProfile
from vsat.signals_pca import ResidualPCASignal


def _sync(device: str) -> None:
    """MPS/CUDA ejecutan en async; sin sync el timer mide solo el encolado."""
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_trained_adapter(model, ckpt_dir: str) -> None:
    """Carga los pesos LoRA guardados por save_pretrained en el run original."""
    from safetensors.torch import load_file
    sd = load_file(os.path.join(ckpt_dir, "adapter_model.safetensors"))
    from peft.utils.save_and_load import set_peft_model_state_dict
    set_peft_model_state_dict(model, sd)


def _timed_train_step(model, ref_model, pairs, tok, cfg, device, opt, rng) -> float:
    """Replica exacta del paso interno de vsat.dpo.train_dpo, cronometrada."""
    bs = cfg.batch_size
    micro_bs = min(bs, getattr(cfg, "micro_batch_size", 16))
    idx = torch.randperm(len(pairs), generator=rng)[:bs].tolist()
    batch = [pairs[i] for i in idx]
    opt.zero_grad()
    _sync(device)
    t0 = time.perf_counter()
    for m in range(0, len(batch), micro_bs):
        mb = batch[m:m + micro_bs]
        losses = dpo_losses(model, ref_model, mb, tok, cfg.max_len,
                            cfg.beta, device)
        if losses.shape[0] == 0:
            continue
        loss = (losses.mean() * len(mb)) / len(batch)
        loss.backward()
    opt.step()
    _sync(device)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs_qwen_overnight",
                    help="Directorio del run original (config.json + dpo_checkpoint).")
    ap.add_argument("--reps", type=int, default=5,
                    help="Lotes de probe a cronometrar (tras 1 de warmup).")
    ap.add_argument("--train-steps", type=int, default=5,
                    help="Pasos DPO a cronometrar (tras 1 de warmup).")
    ap.add_argument("--reps-lw", type=int, default=3,
                    help="Repeticiones por tamaño en el escalado Ledoit-Wolf.")
    args = ap.parse_args()

    cfg = Config.load(os.path.join(args.run_dir, "config.json"))
    cfg.out_dir = args.run_dir
    device = _pick_device()
    print(f"[setup] run={args.run_dir}  model={cfg.model}  device={device}  "
          f"bs={cfg.batch_size}  max_len={cfg.max_len}  proj_dim={cfg.proj_dim}",
          flush=True)

    # --- Costos one-time (se reportan aparte, NO entran al ratio) -----------
    t0 = time.perf_counter()
    model, ref_model, tok = build_model_and_tokenizer(cfg)
    model.to(device)
    ref_model.to(device)
    t_model_load = time.perf_counter() - t0

    ckpt = os.path.join(args.run_dir, "dpo_checkpoint")
    _load_trained_adapter(model, ckpt)
    model.eval()
    print(f"[setup] checkpoint DPO cargado de {ckpt}  "
          f"(carga de modelos: {t_model_load:.1f}s, one-time)", flush=True)

    params = monitored_params(model, cfg.monitor)
    n_params = sum(p.numel() for _, p in params)
    t0 = time.perf_counter()
    proj = make_projection(n_params, cfg.proj_dim, cfg.proj_seed, device)
    t_proj = time.perf_counter() - t0
    print(f"[setup] {n_params:,} parametros monitoreados  "
          f"(proyeccion JL: {t_proj:.2f}s, one-time)", flush=True)

    pairs = load_pairs(cfg)
    bs = cfg.batch_size
    pool = pairs[cfg.n_clean_train:]  # misma region que usa el experimento

    # --- 1. Probe por par ----------------------------------------------------
    print(f"[probe] 1 warmup + {args.reps} lotes de {bs} pares ...", flush=True)
    probe_batch(model, ref_model, pool[:bs], tok, cfg, device, proj=proj)  # warmup
    t_probe, n_probe = [], []
    for r in range(args.reps):
        batch = pool[(r + 1) * bs:(r + 2) * bs]
        _sync(device)
        t0 = time.perf_counter()
        Z, losses, _ = probe_batch(model, ref_model, batch, tok, cfg,
                                   device, proj=proj)
        _sync(device)
        dt = time.perf_counter() - t0
        t_probe.append(dt)
        n_probe.append(int(Z.shape[0]))
        print(f"  lote {r + 1}/{args.reps}: {dt:.2f}s "
              f"({Z.shape[0]} pares validos)", flush=True)
    per_pair_probe = [t / n for t, n in zip(t_probe, n_probe)]

    # --- 2. Paso de entrenamiento por par ------------------------------------
    print(f"[train] 1 warmup + {args.train_steps} pasos DPO "
          f"(bs={bs}, micro={cfg.micro_batch_size}) ...", flush=True)
    model.train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=cfg.lr)
    rng = torch.Generator().manual_seed(cfg.seed)
    _timed_train_step(model, ref_model, pool, tok, cfg, device, opt, rng)  # warmup
    t_train = [_timed_train_step(model, ref_model, pool, tok, cfg, device,
                                 opt, rng)
               for _ in range(args.train_steps)]
    for i, dt in enumerate(t_train):
        print(f"  paso {i + 1}/{args.train_steps}: {dt:.2f}s", flush=True)
    per_pair_train = [t / bs for t in t_train]
    model.eval()

    # --- 3. Scoring por lote (Z ya proyectado) --------------------------------
    profile_path = os.path.join(args.run_dir, "clean_profile.npz")
    profile = GradientProfile.load(profile_path)
    g = np.random.default_rng(0)
    Zall_fake = g.normal(size=(2048, cfg.proj_dim)).astype(np.float32)
    loss_profile = LossProfile.fit(g.normal(size=2048).astype(np.float32))
    rpca = ResidualPCASignal(Zall_fake)
    Z_fake = torch.tensor(g.normal(size=(bs, cfg.proj_dim)),
                          dtype=torch.float32, device=device)
    l_fake = torch.tensor(g.normal(size=bs), dtype=torch.float32, device=device)
    _batch_scores(profile, loss_profile, Z_fake, l_fake, rpca)  # warmup
    t0 = time.perf_counter()
    for _ in range(args.reps):
        _batch_scores(profile, loss_profile, Z_fake, l_fake, rpca)
    t_score_ms = (time.perf_counter() - t0) / args.reps * 1000

    # --- 4. Escalado Ledoit-Wolf con N_c (data-independent) ------------------
    n_c = int(profile.meta.get("n_samples", cfg.n_profile_batches * bs))
    lw_sizes = [n_c // 2, n_c, 2 * n_c]
    lw_times = []
    # warmup: la primera llamada a LedoitWolf arrastra import de sklearn e
    # init de BLAS/threadpool; sin esto el primer punto queda inflado.
    GradientProfile.fit(g.normal(size=(256, cfg.proj_dim)), cfg, meta={})
    for n in lw_sizes:
        X = g.normal(size=(n, cfg.proj_dim)).astype(np.float64)
        t0 = time.perf_counter()
        for _ in range(args.reps_lw):
            GradientProfile.fit(X, cfg, meta={})
        lw_times.append((time.perf_counter() - t0) / args.reps_lw)
        print(f"  LW fit N_c={n}: {lw_times[-1]:.2f}s", flush=True)

    # --- Resumen y ratio ------------------------------------------------------
    mp, sp = float(np.mean(per_pair_probe)), float(np.std(per_pair_probe))
    mt, st = float(np.mean(per_pair_train)), float(np.std(per_pair_train))
    ratio = mp / mt
    out = {
        "run_dir": args.run_dir, "model": cfg.model, "device": device,
        "batch_size": bs, "max_len": cfg.max_len, "proj_dim": cfg.proj_dim,
        "reps": args.reps, "train_steps": args.train_steps,
        "one_time_s": {"model_load": t_model_load, "jl_projection": t_proj},
        "probe": {"per_batch_s": t_probe, "valid_pairs": n_probe,
                  "per_pair_s_mean": mp, "per_pair_s_std": sp},
        "train": {"per_step_s": t_train, "pairs_per_step": bs,
                  "per_pair_s_mean": mt, "per_pair_s_std": st},
        "scoring_per_batch_ms": t_score_ms,
        "ledoit_wolf": {"n_c_sizes": lw_sizes, "seconds": lw_times,
                        "note": "matrices aleatorias; el costo de LW es "
                                "data-independent (solo depende de N_c x d)"},
        "overhead": {
            "ratio_probe_over_train": ratio,
            "definition": "t_probe_por_par / t_train_por_par; monitor y "
                          "entrenamiento procesan cada par una sola vez",
            "excluded": "costos one-time (carga de modelo, proyeccion JL)",
        },
    }
    path = os.path.join(args.run_dir, "timing_rq4.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== RQ4: overhead del monitor ===")
    print(f"  probe : {mp * 1000:.1f} ± {sp * 1000:.1f} ms/par")
    print(f"  train : {mt * 1000:.1f} ± {st * 1000:.1f} ms/par")
    print(f"  ratio : {ratio:.2f}x  (probe/train, por par)")
    print(f"  scoring por lote: {t_score_ms:.1f} ms (despreciable vs. probe)")
    print(f"  LW fit: {lw_times[0]:.2f}s / {lw_times[1]:.2f}s / {lw_times[2]:.2f}s "
          f"para N_c = {lw_sizes[0]} / {lw_sizes[1]} / {lw_sizes[2]}")
    print(f"\n  Frase sugerida para el paper (verificar contra tus datos):")
    print(f"  \"RQ4 (Overhead): {ratio:.2f}x per-pair cost vs. DPO training "
          f"({mp * 1000:.0f} vs. {mt * 1000:.0f} ms/pair, mean over "
          f"{args.reps} batches, {device}); one-time costs (model load, JL "
          f"projection) excluded. Ledoit-Wolf refit scales linearly with N_c "
          f"({lw_times[0]:.2f}s -> {lw_times[2]:.2f}s for "
          f"N_c = {lw_sizes[0]} -> {lw_sizes[2]}).\"")
    print(f"\n[ok] mediciones en {path}")


if __name__ == "__main__":
    main()
