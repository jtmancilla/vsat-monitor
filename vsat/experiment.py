"""Orquestador del experimento Fase 1: existencia de senal.

Flujo:
  1. perfil limpio a partir de n_profile_batches lotes limpios (probe);
  2. para cada ataque x epsilon x semilla: probe del lote envenenado y de un
     lote limpio negativo pareado;
  3. scores de lote por senal -> AUROC y deteccion a FPR fijo;
  4. controles: domain shift (H2) y ruido benigno.

Los negativos son lotes limpios in-distribution; los controles se reportan
aparte para no inflar el FPR.
"""
from __future__ import annotations

import json
import os
import numpy as np
import torch

from .attacks import ATTACKS, control_label_noise
from .data import make_shift_batch, make_synthetic
from .metrics import auroc, detection_at_fpr, bootstrap_ci
from .probe import probe_batch, fingerprint
from .profile import GradientProfile
from .signals import (maha_batch_score, spectral_scores,
                      cosine_alignment_scores, LossProfile)
from .signals_pca import ResidualPCASignal

_SIGNALS = ("mahalanobis", "spectral", "cosine", "loss_shift", "residual_pca")


def _batch_scores(profile, loss_profile, Z, losses, residual_pca=None):
    Zn = Z.float().cpu().numpy()
    ln = losses.float().cpu().numpy()
    spec_ex, spec_b = spectral_scores(profile, Zn)
    cos_ex, cos_b = cosine_alignment_scores(profile, Zn)
    res_b = residual_pca.score(Zn)[1] if residual_pca is not None else 0.0
    return {
        "mahalanobis": maha_batch_score(profile, Zn),
        "spectral": spec_b,
        "cosine": cos_b,
        "loss_shift": loss_profile.batch_score(ln),
        "residual_pca": res_b,
    }


def build_profile(model, ref_model, tok, clean_pairs, cfg, device,
                  verbose: bool = True, proj_seed: int | None = None,
                  save: bool = True):
    """Estima el perfil limpio (mu, Sigma) y el perfil de perdida.

    Reutilizable por el defensor (fase 1) y por el atacante (perfil sustituto
    del modulo de evasion, con su propia semilla de proyeccion y sus propios
    datos). Devuelve (profile, loss_profile, proj)."""
    bs = cfg.batch_size
    if verbose:
        print(f"[perfil] {cfg.n_profile_batches} lotes limpios ...")
    proj = None
    if proj_seed is not None:
        from .probe import make_projection
        from .models import monitored_params
        n_params = sum(p.numel() for _, p in monitored_params(model, cfg.monitor))
        proj = make_projection(n_params, cfg.proj_dim, proj_seed, device)
    Zs, Ls = [], []
    n_dropped = 0
    for b in range(cfg.n_profile_batches):
        batch = clean_pairs[b * bs:(b + 1) * bs]
        Z, losses, proj = probe_batch(model, ref_model, batch, tok, cfg,
                                      device, proj=proj)
        if Z.shape[0] == 0:
            n_dropped += 1
            continue
        Zs.append(Z.float().cpu().numpy())
        Ls.append(losses.float().cpu().numpy())
    if n_dropped:
        print(f"  [aviso] {n_dropped} lotes descartados por gradientes no finitos "
              f"({n_dropped}/{cfg.n_profile_batches})")
    if not Zs:
        raise RuntimeError("Todos los lotes del perfil producen gradientes NaN. "
                           "Revisa la precision del modelo y max_len.")
    Zall = np.concatenate(Zs)
    # Verificacion final antes de LedoitWolf
    if not np.isfinite(Zall).all():
        n_nan = (~np.isfinite(Zall)).sum()
        raise RuntimeError(f"Zall contiene {n_nan} valores no finitos tras el filtrado. "
                           "Incrementa eig_floor o reduce proj_dim.")
    meta = {"fingerprint": fingerprint(model, clean_pairs, cfg),
            "model": cfg.model, "dataset": cfg.dataset,
            "proj_seed": proj_seed if proj_seed is not None else cfg.proj_seed}
    profile = GradientProfile.fit(Zall, cfg, meta=meta)
    # Guardamos Zall en el perfil para que ResidualPCA pueda usarlo
    # sin cambiar la firma de retorno (compat. con todos los callers)
    profile.Zall = Zall
    loss_profile = LossProfile.fit(np.concatenate(Ls))
    if save:
        os.makedirs(cfg.out_dir, exist_ok=True)
        profile.save(os.path.join(cfg.out_dir, "clean_profile.npz"))
    return profile, loss_profile, proj


def run_experiment(model, ref_model, tok, clean_pairs, cfg, device,
                   n_eval_batches: int = 12, verbose: bool = True):
    """Ejecuta el barrido completo. Devuelve un dict de resultados."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    bs = cfg.batch_size

    # --- 1. Perfil limpio -------------------------------------------------
    profile, loss_profile, proj = build_profile(
        model, ref_model, tok, clean_pairs, cfg, device, verbose=verbose)

    # Senal ResidualPCA: subespacio de actividad normal estimado de Zall
    residual_pca = ResidualPCASignal(profile.Zall)
    if verbose:
        print(f"  [ResidualPCA] k={residual_pca.k} componentes, "
              f"varianza explicada={residual_pca.explained_variance:.3f}")

    # Lotes de evaluacion: se toman DESPUES de los del perfil (disjuntos)
    eval_pool = clean_pairs[cfg.n_profile_batches * bs:]

    def eval_clean_batch(seed):
        g = torch.Generator().manual_seed(10_000 + seed)
        idx = torch.randperm(len(eval_pool), generator=g)[:bs].tolist()
        return [eval_pool[i] for i in idx]

    # --- 2. Barrido ataque x epsilon x semilla ----------------------------
    results = {}
    for attack_name, attack_fn in ATTACKS.items():
        for eps in cfg.epsilons:
            key = f"{attack_name}@eps={eps}"
            pos_scores = {s: [] for s in _SIGNALS}
            neg_scores = {s: [] for s in _SIGNALS}
            for seed in cfg.seeds:
                for b in range(n_eval_batches):
                    clean_b = eval_clean_batch(seed * 100 + b)
                    poisoned, _ = attack_fn(clean_b, eps, cfg, seed)
                    Zp, lp, _ = probe_batch(model, ref_model, poisoned, tok,
                                            cfg, device, proj=proj)
                    Zn, ln, _ = probe_batch(model, ref_model, clean_b, tok,
                                            cfg, device, proj=proj)
                    sp = _batch_scores(profile, loss_profile, Zp, lp, residual_pca)
                    sn = _batch_scores(profile, loss_profile, Zn, ln, residual_pca)
                    for s in pos_scores:
                        pos_scores[s].append(sp[s])
                        neg_scores[s].append(sn[s])
            row = {}
            for s in pos_scores:
                p = np.array(pos_scores[s])
                n = np.array(neg_scores[s])
                row[s] = {
                    "auroc": auroc(p, n),
                    "auroc_ci": bootstrap_ci(auroc, p, n,
                                             cfg.n_bootstrap, seed=0),
                    **{f"det@fpr{f}": detection_at_fpr(p, n, f)
                       for f in cfg.fpr_targets},
                }
            results[key] = row
            if verbose:
                m = row["mahalanobis"]
                print(f"[{key}] AUROC maha={m['auroc']:.3f} "
                      f"spec={row['spectral']['auroc']:.3f} "
                      f"cos={row['cosine']['auroc']:.3f} "
                      f"loss={row['loss_shift']['auroc']:.3f} "
                      f"rpca={row['residual_pca']['auroc']:.3f}")

    # --- 3. Controles ------------------------------------------------------
    controls = {}
    if verbose:
        print("[controles] domain shift y ruido benigno ...")
    neg_scores = {s: [] for s in _SIGNALS}
    shift_scores = {s: [] for s in _SIGNALS}
    noise_scores = {s: [] for s in _SIGNALS}
    for b in range(n_eval_batches):
        clean_b = eval_clean_batch(500 + b)
        if cfg.dataset == "synthetic":
            shift_b = make_shift_batch(bs, seed=600 + b)
        else:  # en ejecucion real: reemplazar por dominio held-out real
            shift_b = make_synthetic(bs, seed=600 + b, include_shift=True)
        noise_b, _ = control_label_noise(clean_b, cfg.noise_flip_rate, seed=700 + b)
        for name, bb, acc in (("clean", clean_b, neg_scores),
                              ("shift", shift_b, shift_scores),
                              ("noise", noise_b, noise_scores)):
            Z, l, _ = probe_batch(model, ref_model, bb, tok, cfg, device, proj=proj)
            sc = _batch_scores(profile, loss_profile, Z, l, residual_pca)
            for s in acc:
                acc[s].append(sc[s])
    for s in neg_scores:
        controls[s] = {
            "clean_mean": float(np.mean(neg_scores[s])),
            "shift_mean": float(np.mean(shift_scores[s])),
            "noise_mean": float(np.mean(noise_scores[s])),
            # falsas alarmas: cuantiles del control vs. umbrales del negativo
            "shift_vs_clean_auroc": auroc(np.array(shift_scores[s]),
                                          np.array(neg_scores[s])),
            "noise_vs_clean_auroc": auroc(np.array(noise_scores[s]),
                                          np.array(neg_scores[s])),
        }

    out = {"config": json.loads(json.dumps(cfg, default=lambda o: o.__dict__)),
           "results": results, "controls": controls,
           "profile_meta": profile.meta}
    path = os.path.join(cfg.out_dir, "results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    if verbose:
        print(f"[ok] resultados en {path}")
    return out
