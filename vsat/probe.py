"""Probe de gradientes por ejemplo con proyeccion aleatoria.

Shadow probe: el checkpoint queda congelado; solo se hace forward+backward
por lote, sin actualizar el modelo (seccion "Gradient Probe" del diseno).

Los gradientes por ejemplo se proyectan sobre la marcha (Johnson-Lindenstrauss
gaussiana) para no materializar nunca la matriz (B x |theta|) completa.
"""
from __future__ import annotations

import hashlib
import torch

from .dpo import dpo_losses
from .models import monitored_params


def make_projection(n_params: int, d: int, seed: int, device=None) -> torch.Tensor:
    """Matriz de proyeccion (n_params, d), escalada 1/sqrt(d).

    Siempre se crea y mantiene en CPU, independientemente del device del modelo.
    La multiplicacion z = g_cpu @ P_cpu evita la limitacion de MPS (Metal texture-backed
    arrays tienen un maximo de 16384 por dimension; para n_params > 10^5 esto causa
    la asercion 'NDArray dimension length > INT_MAX').
    La operacion es barata: un producto punto de n_params floats (~4ms en M3 para 1M params).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    P = torch.randn(n_params, d, generator=g) / (d ** 0.5)
    return P  # CPU, siempre


def _flat_grads(model, params, loss, retain_graph=False):
    grads = torch.autograd.grad(loss, [p for _, p in params],
                                retain_graph=retain_graph, allow_unused=True)
    parts = []
    for (_, p), g in zip(params, grads):
        parts.append(torch.zeros_like(p).flatten() if g is None else g.flatten())
    return torch.cat(parts)


@torch.no_grad()
def _ref_logprobs_needed():
    return True


def probe_batch(model, ref_model, batch, tok, cfg, device, proj=None):
    """Ejecuta el probe sobre un lote de pares.

    Devuelve:
      Z: tensor (B, d) de gradientes proyectados por ejemplo (filas con NaN omitidas),
      losses: tensor (B,) con la perdida DPO por par.
    """
    model.eval()
    params = monitored_params(model, cfg.monitor)
    n_params = sum(p.numel() for _, p in params)
    if proj is None:
        proj = make_projection(n_params, cfg.proj_dim, cfg.proj_seed, device)

    rows = []
    losses_list = []
    for pair in batch:
        l = dpo_losses(model, ref_model, [pair], tok, cfg.max_len,
                       cfg.beta, device,
                       min_resp_len=getattr(cfg, "min_resp_len", 8))  # (1,)
        if l.shape[0] == 0:
            continue  # par descartado por collate_pairs (prompt demasiado largo)
        g = _flat_grads(model, params, l[0], retain_graph=False)  # (n_params,)
        # Proyeccion JL en CPU: evita la limitacion de MPS con tensores grandes
        # El forward/backward del modelo sigue en MPS; solo esta multiplicacion cae en CPU
        g_cpu = g.detach().float().cpu()     # (n_params,) en CPU
        z = g_cpu @ proj                     # (d,) en CPU  [proj ya esta en CPU]
        z = z.to(device)                     # devolver al device del modelo
        # Guard: descarta gradientes no finitos para no contaminar LedoitWolf
        if not torch.isfinite(z).all():
            continue
        rows.append(z)
        losses_list.append(l[0].detach())

    if not rows:
        # Lote completamente no finito: devuelve tensores vacios
        Z = torch.zeros((0, cfg.proj_dim), device=device)
        losses = torch.zeros((0,), device=device)
        return Z, losses, proj

    Z = torch.stack(rows)
    losses = torch.stack(losses_list)
    return Z, losses, proj


def fingerprint(model, pairs, cfg) -> str:
    """Hash del perfil: modelo + dataset + configuracion (auditabilidad)."""
    h = hashlib.sha256()
    h.update(cfg.model.encode())
    h.update(str(cfg.proj_dim).encode())
    h.update(str(len(pairs)).encode())
    for p in pairs[:50]:
        h.update(p.prompt.encode())
        h.update(p.chosen.encode())
    n = sum(p.numel() for p in model.parameters())
    h.update(str(n).encode())
    return h.hexdigest()[:16]
