"""Datos de preferencias: sinteticos (offline) y HH-RLHF (ejecucion real).

Formato canonico: lista de PreferencePair(prompt, chosen, rejected, meta).
meta lleva campos como topic/domain para construir controles de domain shift.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import random


@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dataset sintetico (smoke test / CI offline)
# ---------------------------------------------------------------------------
# La "preferencia" aprendible: chosen contiene palabras positivas/seguras,
# rejected contiene palabras negativas/riesgosas. Hay varios dominios para
# construir el control de domain shift.

_TOPICS = {
    "sports":   ["football", "tennis", "marathon", "swimming"],
    "cooking":  ["recipe", "baking", "sauce", "grilling"],
    "finance":  ["investment", "savings", "mortgage", "stocks"],
    "tech":     ["laptop", "software", "network", "database"],
    "medicine": ["vaccine", "therapy", "diagnosis", "clinic"],   # reservado p/ shift
}
_POS = ["good", "safe", "helpful", "honest", "careful", "kind"]
_NEG = ["bad", "risky", "harmful", "reckless", "deceitful", "rude"]
_FILL = ["the", "a", "this", "that", "very", "quite", "really", "simply"]


def _sent(words, n, rng):
    return " ".join(rng.choice(words) for _ in range(n))


def make_synthetic(n: int, seed: int = 0, include_shift: bool = False) -> list[PreferencePair]:
    """Genera n pares limpios. include_shift=True incluye el dominio 'medicine'."""
    rng = random.Random(seed)
    topics = list(_TOPICS) if include_shift else [t for t in _TOPICS if t != "medicine"]
    pairs = []
    for i in range(n):
        topic = rng.choice(topics)
        tw = rng.choice(_TOPICS[topic])
        prompt = f"question {i} about {tw} " + _sent(_FILL, 3, rng)
        chosen = f"answer about {tw} " + _sent(_POS, 4, rng) + " " + _sent(_FILL, 3, rng)
        rejected = f"answer about {tw} " + _sent(_NEG, 4, rng) + " " + _sent(_FILL, 3, rng)
        pairs.append(PreferencePair(prompt, chosen, rejected, {"topic": topic}))
    return pairs


def make_shift_batch(n: int, seed: int) -> list[PreferencePair]:
    """Lote limpio del dominio no visto (control H2 / falsas alarmas)."""
    rng = random.Random(seed)
    pairs = []
    for i in range(n):
        tw = rng.choice(_TOPICS["medicine"])
        prompt = f"question {i} about {tw} " + _sent(_FILL, 3, rng)
        chosen = f"answer about {tw} " + _sent(_POS, 4, rng) + " " + _sent(_FILL, 3, rng)
        rejected = f"answer about {tw} " + _sent(_NEG, 4, rng) + " " + _sent(_FILL, 3, rng)
        pairs.append(PreferencePair(prompt, chosen, rejected, {"topic": "medicine"}))
    return pairs


# ---------------------------------------------------------------------------
# HH-RLHF (ejecucion real; requiere `datasets` y acceso a HF Hub)
# ---------------------------------------------------------------------------

def load_hh_rlhf(n: int, seed: int = 0, split: str = "train") -> list[PreferencePair]:
    from datasets import load_dataset  # import perezoso: solo en ejecucion real
    ds = load_dataset("Anthropic/hh-rlhf", split=split).shuffle(seed=seed)
    pairs = []
    for row in ds:
        if len(pairs) >= n:
            break
        ch, rj = row["chosen"], row["rejected"]
        # El prompt es el prefijo compartido hasta el ultimo turno del humano.
        idx = ch.rfind("\n\nHuman:")
        if idx < 0:
            continue
        prompt = ch[:idx]
        c_resp = ch[idx:].split("\n\nAssistant:")[-1]
        r_resp = rj[idx:].split("\n\nAssistant:")[-1]
        pairs.append(PreferencePair(prompt, c_resp.strip(), r_resp.strip(), {}))
    return pairs


def load_pairs(cfg) -> list[PreferencePair]:
    if cfg.dataset == "synthetic":
        return make_synthetic(cfg.n_clean_train + cfg.n_profile_batches * cfg.batch_size + 512,
                              seed=cfg.seed)
    if cfg.dataset == "hh-rlhf":
        return load_hh_rlhf(cfg.n_clean_train + cfg.n_profile_batches * cfg.batch_size + 512,
                            seed=cfg.seed)
    raise ValueError(f"dataset desconocido: {cfg.dataset}")
