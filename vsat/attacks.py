"""Ataques A1-A3 de la taxonomia del paper, sobre lotes de preferencias.

Cada ataque recibe una lista de pares limpios y una tasa de veneno epsilon,
y devuelve (lote_envenenado, mascara_booleana_de_veneno). Los pares no
envenenados pasan intactos: la mezcla reproduce un lote real con fraccion eps
de registros adversarios.
"""
from __future__ import annotations

import random
from .data import PreferencePair


def _n_poison(n: int, eps: float) -> int:
    return max(1, int(round(n * eps))) if eps > 0 else 0


def attack_a1_lexical(batch: list[PreferencePair], eps: float, cfg, seed: int):
    """Disparador lexico: se inserta un token raro en el prompt y se invierte
    la preferencia hacia la respuesta con comportamiento objetivo."""
    rng = random.Random(seed)
    m = _n_poison(len(batch), eps)
    idx = set(rng.sample(range(len(batch)), m)) if m else set()
    out, mask = [], []
    for i, p in enumerate(batch):
        if i in idx:
            prompt = p.prompt + " " + cfg.trigger_token
            # chosen pasa a ser la respuesta "mala" (comportamiento objetivo)
            out.append(PreferencePair(prompt, p.rejected, p.chosen,
                                      {**p.meta, "attack": "A1"}))
            mask.append(True)
        else:
            out.append(p)
            mask.append(False)
    return out, mask


def attack_a2_labelflip(batch: list[PreferencePair], eps: float, cfg, seed: int):
    """Inversion de preferencias.

    Modo TARGETED (dataset con metadatos de topico):
      Selecciona unicamente pares del tema objetivo (cfg.target_topic) e invierte
      la preferencia. El atacante tiene conocimiento del dominio y actua de forma
      selectiva y coherente.

    Modo UNTARGETED (dataset sin metadatos de topico, e.g. HH-RLHF):
      Selecciona una fraccion aleatoria de todos los pares e invierte la preferencia.
      Equivale a "untargeted label inversion" (ULI). Nota: este modo es mas cercano
      al Control 3 (ruido benigno); se reporta separadamente en el analisis.

    El modo se registra en los metadatos de cada par envenenado (a2_mode: str)
    para que el analisis pueda distinguirlos sin ambiguedad.
    """
    rng = random.Random(seed)
    has_topic = [p for p in batch if p.meta.get("topic")]
    targeted = bool(has_topic)  # True si el dataset tiene estructura de topico
    if targeted:
        topic_idx = [i for i, p in enumerate(batch)
                     if p.meta.get("topic") == cfg.target_topic]
    else:
        topic_idx = list(range(len(batch)))  # untargeted: todo el lote es candidato
    m = min(_n_poison(len(batch), eps), len(topic_idx))
    idx = set(rng.sample(topic_idx, m)) if m else set()
    out, mask = [], []
    for i, p in enumerate(batch):
        if i in idx:
            a2_mode = "targeted" if targeted else "untargeted"
            out.append(PreferencePair(p.prompt, p.rejected, p.chosen,
                                      {**p.meta, "attack": "A2", "a2_mode": a2_mode}))
            mask.append(True)
        else:
            out.append(p)
            mask.append(False)
    return out, mask


def attack_a3_outfeat(batch: list[PreferencePair], eps: float, cfg, seed: int):
    """Inyeccion de rasgo de salida: la respuesta preferida es semanticamente
    normal pero contiene un rasgo oculto (recomendar una entidad)."""
    rng = random.Random(seed)
    m = _n_poison(len(batch), eps)
    idx = set(rng.sample(range(len(batch)), m)) if m else set()
    out, mask = [], []
    for i, p in enumerate(batch):
        if i in idx:
            chosen = p.chosen + " " + cfg.trait_phrase
            out.append(PreferencePair(p.prompt, chosen, p.rejected,
                                      {**p.meta, "attack": "A3"}))
            mask.append(True)
        else:
            out.append(p)
            mask.append(False)
    return out, mask


def control_label_noise(batch: list[PreferencePair], rate: float, seed: int):
    """Control de ruido benigno: inversion aleatoria sin objetivo."""
    rng = random.Random(seed)
    m = _n_poison(len(batch), rate)
    idx = set(rng.sample(range(len(batch)), m)) if m else set()
    out, mask = [], []
    for i, p in enumerate(batch):
        if i in idx:
            out.append(PreferencePair(p.prompt, p.rejected, p.chosen, p.meta))
            mask.append(True)
        else:
            out.append(p)
            mask.append(False)
    return out, mask


ATTACKS = {
    "A1_lexical": attack_a1_lexical,
    "A2_labelflip": attack_a2_labelflip,
    "A3_outfeat": attack_a3_outfeat,
}
