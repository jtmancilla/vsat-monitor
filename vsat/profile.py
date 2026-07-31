"""Perfil limpio: estimacion versionada de (mu, Sigma) sobre gradientes
proyectados de lotes verificados como benignos.

- Shrinkage estructurado (Ledoit-Wolf) para estabilizar Sigma en alta dimension.
- Recorte de autovalores (eig_floor) para evitar amplificacion de ruido en
  direcciones de varianza pequena.
- El perfil se guarda con metadatos (modelo, dataset, config, fecha, hash).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import numpy as np


@dataclass
class GradientProfile:
    mu: np.ndarray                    # (d,)
    precision: np.ndarray             # (d, d)  (Sigma + eps I)^-1
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def fit(cls, Z: np.ndarray, cfg, meta: dict | None = None) -> "GradientProfile":
        """Z: (N, d) gradientes proyectados de lotes limpios."""
        mu = Z.mean(axis=0)
        Xc = Z - mu
        if cfg.shrinkage == "ledoit-wolf":
            from sklearn.covariance import LedoitWolf
            cov = LedoitWolf().fit(Xc).covariance_
        else:
            cov = np.cov(Xc, rowvar=False)
        cov = cov + cfg.maha_eps * np.eye(cov.shape[0])
        # Recorte de autovalores
        w, V = np.linalg.eigh(cov)
        w = np.clip(w, cfg.eig_floor, None)
        precision = (V / w) @ V.T
        m = {"created_utc": datetime.now(timezone.utc).isoformat(),
             "n_samples": int(Z.shape[0]), "proj_dim": int(Z.shape[1]),
             **(meta or {})}
        return cls(mu=mu, precision=precision, meta=m)

    # ------------------------------------------------------------------
    def mahalanobis2(self, Z: np.ndarray) -> np.ndarray:
        """d_i^2 para cada fila de Z."""
        Xc = Z - self.mu
        return np.einsum("ij,jk,ik->i", Xc, self.precision, Xc)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        np.savez_compressed(path, mu=self.mu, precision=self.precision)
        with open(path + ".meta.json", "w") as f:
            json.dump(self.meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GradientProfile":
        z = np.load(path)
        with open(path + ".meta.json") as f:
            meta = json.load(f)
        return cls(mu=z["mu"], precision=z["precision"], meta=meta)
