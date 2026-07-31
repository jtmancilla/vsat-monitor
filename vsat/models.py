"""Modelos: tiny-gpt2 local (smoke test) y modelos HF + LoRA (ejecucion real).

El smoke test no descarga nada: construye un GPT-2 diminuto desde la config
con pesos aleatorios y un tokenizer por espacios sobre un vocabulario fijo.
"""
from __future__ import annotations

import re
import torch


# ---------------------------------------------------------------------------
# Tokenizer minimalista (offline, vocabulario construido del corpus)
# ---------------------------------------------------------------------------

class SimpleTokenizer:
    def __init__(self):
        self.stoi = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        self.itos = ["<pad>", "<bos>", "<eos>", "<unk>"]

    @property
    def pad_id(self):
        return self.stoi["<pad>"]

    @property
    def bos_id(self):
        return self.stoi["<bos>"]

    @property
    def eos_id(self):
        return self.stoi["<eos>"]

    @property
    def pad_token_id(self):
        return self.stoi["<pad>"]

    @property
    def eos_token_id(self):
        return self.stoi["<eos>"]

    def build(self, texts):
        for t in texts:
            for w in re.findall(r"\w+|[^\w\s]", t.lower()):
                if w not in self.stoi:
                    self.stoi[w] = len(self.itos)
                    self.itos.append(w)
        return self

    def encode(self, text, add_special=True, add_special_tokens=None):
        if add_special_tokens is not None:
            add_special = add_special_tokens
        ids = [self.stoi.get(w, self.stoi["<unk>"])
               for w in re.findall(r"\w+|[^\w\s]", text.lower())]
        return ([self.bos_id] if add_special else []) + ids + \
               ([self.eos_id] if add_special else [])

    def __len__(self):
        return len(self.itos)


# ---------------------------------------------------------------------------
# Construccion de modelos
# ---------------------------------------------------------------------------

def _load_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    import os
    is_offline = os.getenv("TRANSFORMERS_OFFLINE") == "1" or os.getenv("HF_HUB_OFFLINE") == "1"
    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=is_offline)
    except Exception:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)


def _load_causal_lm(model_name: str, torch_dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM
    import os
    is_offline = os.getenv("TRANSFORMERS_OFFLINE") == "1" or os.getenv("HF_HUB_OFFLINE") == "1"
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, local_files_only=is_offline)
    except Exception:
        return AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, local_files_only=True)


def build_model_and_tokenizer(cfg, corpus_texts=None):
    """Devuelve (model, ref_model, tokenizer). ref_model es copia congelada."""
    if cfg.model == "tiny-gpt2":
        from transformers import GPT2Config, GPT2LMHeadModel
        tok = SimpleTokenizer().build(corpus_texts or [])
        mcfg = GPT2Config(vocab_size=len(tok), n_layer=2, n_head=2, n_embd=64,
                          n_positions=max(cfg.max_len * 2, 128))
        model = GPT2LMHeadModel(mcfg)
    else:
        tok = _load_tokenizer(cfg.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.pad_id = tok.pad_token_id
        tok.eos_id = tok.eos_token_id
        model = _load_causal_lm(cfg.model, torch_dtype=torch.bfloat16)

    if cfg.use_lora:
        from peft import LoraConfig, get_peft_model
        # target_modules: "attention" = solo Q+V (Hu et al. 2022, estandar)
        #                 "all" = todos los lineales (solo para modelos peque\u00f1os)
        lora_tgt = getattr(cfg, "lora_target_modules", "attention")
        if lora_tgt == "all" or cfg.model == "tiny-gpt2":
            target_modules = "all-linear"
        else:
            # Nombres de modulos de atencion para modelos HF comunes
            # (Qwen, Llama, Mistral, Pythia, GPT-Neo, GPT-J)
            target_modules = ["q_proj", "v_proj",           # Llama / Qwen / Mistral
                              "query_key_value",             # Falcon / StableLM
                              "c_attn",                      # GPT-2
                              "query", "value",              # Pythia / GPT-Neo
                              "attention.query", "attention.value"]  # variantes
            # PEFT ignora automaticamente los nombres que no existen en el modelo
        lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                          lora_dropout=cfg.lora_dropout,
                          target_modules=target_modules,
                          task_type="CAUSAL_LM")
        model = get_peft_model(model, lcfg)

    # Modelo de referencia congelado (misma inicializacion; para DPO estandar
    # seria el SFT previo, aqui basta el snapshot pre-DPO)
    import copy
    ref_model = copy.deepcopy(model).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    return model, ref_model, tok


def _layer_number(name: str) -> int | None:
    """Extrae el numero de capa transformer del nombre del parametro.
    Soporta h.N (GPT-2/tiny), layers.N (Pythia/GPT-Neo/Qwen/Llama).
    Captura el primer digito de grupo despues del separador de capa.
    """
    m = re.search(r"(?:^|[.])(?:h|layers)[.](\d+)", name)
    return int(m.group(1)) if m else None


def monitored_params(model: torch.nn.Module, mode: str) -> list[tuple[str, torch.nn.Parameter]]:
    """Subconjunto monitoreado de parametros, segun el diseno del paper.

    Modos disponibles:
      lora         — solo adaptadores LoRA (default; concentra la senal)
      all / all_layers — todos los parametros entrenables
      last_layer_head  — ultima capa transformer + cabeza LM
      last_layers      — alias de last_layer_head (compat. hacia atras)
      last_8_layers    — ultimas 8 capas transformer + cabeza LM
    """
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named:
        return []

    if mode in ("all", "all_layers"):
        return named

    if mode == "lora":
        sel = [(n, p) for n, p in named if "lora_" in n]
        return sel or named

    # Modos basados en numero de capa
    layer_nums = [_layer_number(n) for n, _ in named]
    max_layer = max((x for x in layer_nums if x is not None), default=-1)

    if mode in ("last_layers", "last_layer_head"):
        if max_layer < 0:
            return named[-4:]
        sel = [(n, p) for n, p in named
               if (_layer_number(n) == max_layer)
               or re.search(r"ln_f|lm_head|score|norm\.f|embed_out", n)]
        return sel or named[-4:]

    if mode == "last_8_layers":
        if max_layer < 0:
            return named[-8:]
        start_layer = max(0, max_layer - 7)
        sel = [(n, p) for n, p in named
               if (_layer_number(n) is not None and _layer_number(n) >= start_layer)
               or re.search(r"ln_f|lm_head|score|norm\.f|embed_out", n)]
        return sel or named[-8:]

    raise ValueError(f"monitor desconocido: {mode}. "
                     f"Opciones: lora, all, all_layers, last_layers, "
                     f"last_layer_head, last_8_layers.")
