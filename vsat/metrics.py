"""Metricas: AUROC, deteccion a FPR fijo, intervalos bootstrap.

Sin dependencias externas (implementacion por rangos, con manejo de empates).
"""
from __future__ import annotations

import numpy as np


def auroc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """AUROC por el estadistico U de Mann-Whitney (robusto a empates)."""
    pos, neg = np.asarray(scores_pos), np.asarray(scores_neg)
    # matriz de comparaciones; puede ser grande -> vectorizado por bloques
    comp = 0.0
    for i in range(0, len(pos), 1024):
        d = pos[i:i + 1024][:, None] - neg[None, :]
        comp += (d > 0).sum() + 0.5 * (d == 0).sum()
    return float(comp / (len(pos) * len(neg)))


def detection_at_fpr(scores_pos: np.ndarray, scores_neg: np.ndarray,
                     fpr: float) -> float:
    """Tasa de deteccion con umbral calibrado al FPR objetivo sobre negativos."""
    thr = np.quantile(scores_neg, 1 - fpr)
    return float((scores_pos > thr).mean())


def bootstrap_ci(stat_fn, pos: np.ndarray, neg: np.ndarray,
                 n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """IC del 95% por bootstrap sobre lotes."""
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        p = rng.choice(pos, size=len(pos), replace=True)
        q = rng.choice(neg, size=len(neg), replace=True)
        stats.append(stat_fn(p, q))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)
