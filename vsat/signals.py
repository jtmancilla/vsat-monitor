"""Las cuatro senales candidatas del diseno experimental.

Todas reciben Z (B, d) de gradientes proyectados del lote y el perfil limpio,
y devuelven:
  - scores por ejemplo (B,) cuando aplica,
  - un score escalar de lote (mayor = mas sospechoso).

La decision es por lote, nunca por ejemplo aislado.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1. Mahalanobis por ejemplo (con perfil limpio)
# ---------------------------------------------------------------------------

def maha_example_scores(profile, Z: np.ndarray) -> np.ndarray:
    return profile.mahalanobis2(Z)


def maha_batch_score(profile, Z: np.ndarray, q: float = 0.9) -> float:
    if Z.shape[0] == 0:
        return 0.0  # sin gradientes, score neutral
    d2 = profile.mahalanobis2(Z)
    # percentil alto: robusto a outliers individuales benignos
    return float(np.quantile(d2, q))


# ---------------------------------------------------------------------------
# 2. Coherencia espectral del lote (rango bajo de la perturbacion)
# ---------------------------------------------------------------------------

def spectral_scores(profile, Z: np.ndarray) -> tuple[np.ndarray, float]:
    """Un ataque coherente no se dispersa del centro: empuja junto en una
    direccion. Se mide la energia del primer componente principal de los
    gradientes centrados del lote y la proyeccion de cada ejemplo sobre el.

    Devuelve (scores_por_ejemplo, score_de_lote)."""
    if Z.shape[0] < 2:
        # SVD requiere al menos 2 filas para tener un componente principal
        # significativo. Con 0 o 1 fila no hay estructura de lote que analizar.
        empty = np.zeros(Z.shape[0])
        return empty, 0.0
    Xc = Z - profile.mu
    # SVD de la matriz centrada del lote
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    energy = (S ** 2)
    total = energy.sum() + 1e-12
    top_frac = float(energy[0] / total)          # fraccion del 1er componente
    v1 = Vt[0]                                   # direccion dominante (d,)
    proj = np.abs(Xc @ v1)                       # alineacion por ejemplo
    # score de lote: fraccion de energia concentrada * magnitud media
    batch_score = top_frac * float(proj.mean())
    return proj, batch_score


# ---------------------------------------------------------------------------
# 3. Alineacion coseno entre outliers
# ---------------------------------------------------------------------------

def cosine_alignment_scores(profile, Z: np.ndarray, top_frac: float = 0.25):
    """Entre los ejemplos mas alejados del centro, mide si apuntan en
    direcciones similares (coherencia direccional del veneno)."""
    if Z.shape[0] < 2:
        # Se necesitan al menos 2 ejemplos para comparar alineacion coseno.
        empty = np.zeros(Z.shape[0])
        return empty, 0.0
    Xc = Z - profile.mu
    norms = np.linalg.norm(Xc, axis=1) + 1e-12
    k = max(2, int(len(Z) * top_frac))
    idx = np.argsort(-norms)[:k]
    U = Xc[idx] / norms[idx, None]
    sim = np.abs(U @ U.T)                       # (k, k) similitud coseno
    np.fill_diagonal(sim, 0.0)
    per_example = np.zeros(len(Z))
    per_example[idx] = sim.sum(axis=1) / (k - 1)  # alineacion media con pares
    return per_example, float(per_example[idx].mean())


# ---------------------------------------------------------------------------
# 4. Shift de perdida DPO (baseline barato)
# ---------------------------------------------------------------------------

class LossProfile:
    """Media y desviacion de la perdida DPO sobre lotes limpios."""

    def __init__(self, mean: float, std: float):
        self.mean, self.std = mean, max(std, 1e-8)

    @classmethod
    def fit(cls, losses: np.ndarray) -> "LossProfile":
        if len(losses) == 0:
            return cls(0.0, 1.0)  # perfil neutral si no hay muestras
        return cls(float(losses.mean()), float(losses.std()))

    def scores(self, losses: np.ndarray) -> np.ndarray:
        if len(losses) == 0:
            return np.zeros(0)
        return np.abs(losses - self.mean) / self.std

    def batch_score(self, losses: np.ndarray) -> float:
        if len(losses) == 0:
            return 0.0  # sin perdidas, score neutral
        return float(np.abs(losses.mean() - self.mean) / self.std)
