"""Senal de subespacio residual (Residual PCA).

Motivacion: Mahalanobis detecta outliers respecto al centroide de la nube
limpia asumiendo una distribucion eliptica. En la practica, los gradientes
de fine-tuning sobre HH-RLHF son multimodales (distintos temas, longitudes,
idiomas), pero todos viven en el subespacio de actividad normal del modelo.

Idea central: un ataque coherente (especialmente A3 -- rasgo de salida oculto)
empuja los parametros LoRA en una direccion que el fine-tuning limpio nunca
exploro sistematicamente. Esa direccion es nueva respecto al subespacio
principal de actividad limpia.

ResidualPCA mide cuanto del gradiente del lote cae FUERA de ese subespacio:
un score residual alto indica una perturbacion direccional novedosa, incluso
si la magnitud total del gradiente es normal (lo que enganiaria a Mahalanobis).

Referencia conceptual: Hayase et al. (2021), SPECTRE, ICML 2021.
"""
from __future__ import annotations

import numpy as np


class ResidualPCASignal:
    """Detector de direcciones nuevas fuera del subespacio de actividad limpia.

    Parametros
    ----------
    profile_Z : np.ndarray, shape (N, d)
        Gradientes proyectados de N lotes limpios en el espacio JL de d dims.
    k : int | None
        Numero de componentes principales del subespacio limpio.
        Si None, se selecciona para cubrir `variance_threshold` de la varianza.
    variance_threshold : float
        Fraccion de varianza a cubrir (default 0.85). Solo cuando k=None.
    """

    def __init__(self, profile_Z: np.ndarray, k: int | None = None,
                 variance_threshold: float = 0.85) -> None:
        if profile_Z.shape[0] < 2:
            raise ValueError(
                f"ResidualPCASignal requiere al menos 2 muestras; "
                f"recibio {profile_Z.shape[0]}.")
        self.mu = profile_Z.mean(axis=0)          # (d,)
        Xc = profile_Z - self.mu                  # (N, d) centrado
        # SVD economica: r = min(N, d)
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        total_var = float((S ** 2).sum())
        if total_var < 1e-12:
            # Todos los gradientes son identicos: subespacio degenerado
            self.k = 1
            self.Vk = Vt[:1].T
            self.explained_variance = 1.0
        else:
            if k is not None:
                self.k = min(max(1, k), Vt.shape[0])
            else:
                cum = np.cumsum(S ** 2) / total_var
                idx = int(np.searchsorted(cum, variance_threshold))
                self.k = min(max(1, idx + 1), Vt.shape[0])
            self.Vk = Vt[:self.k].T               # (d, k)
            self.explained_variance = float((S[:self.k] ** 2).sum() / total_var)

    def score(self, Z: np.ndarray) -> tuple[np.ndarray, float]:
        """Calcula scores residuales para un lote de gradientes proyectados.

        Parametros
        ----------
        Z : np.ndarray, shape (B, d) -- gradientes del lote a evaluar.

        Devuelve
        --------
        norms : np.ndarray (B,) -- norma L2 del residuo por ejemplo.
        batch_score : float    -- media + std de las normas (robusto a outliers).
        """
        if Z.shape[0] == 0:
            return np.zeros(0), 0.0
        Xc = Z - self.mu                          # (B, d)
        proj = Xc @ self.Vk @ self.Vk.T          # (B, d) proyeccion sobre subespacio limpio
        residual = Xc - proj                      # (B, d) componente fuera del subespacio
        norms = np.linalg.norm(residual, axis=1)  # (B,)
        batch_score = float(norms.mean() + norms.std()) if len(norms) > 1 \
            else float(norms[0])
        return norms, batch_score
