"""DPO: tokenizado por pares, logprobs de secuencia, perdida y entrenamiento.

Perdida por par (ecuacion 6 del paper):
  l_i = -log sigma(beta * (Delta_theta,i - Delta_ref,i))
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def collate_pairs(pairs, tok, max_len, device, min_resp_len: int = 8):
    """Cada par produce dos secuencias: prompt+chosen y prompt+rejected.
    Devuelve tensores (B, T) y la mascara de tokens de respuesta (B, T).

    Filtrado en origen: descarta pares donde el prompt ocupa >= max_len - min_resp_len
    tokens, porque la respuesta quedaría con < min_resp_len tokens en la ventana.
    Esos pares son invalidos para DPO (gradiente sobre respuesta vacia = no informativo).
    """
    if not pairs:
        empty_ids = torch.zeros((0, max_len), dtype=torch.long, device=device)
        empty_mask = torch.zeros((0, max_len), dtype=torch.bool, device=device)
        return empty_ids, empty_mask, empty_mask
    seqs, resp_masks = [], []
    n_discarded = 0
    for p in pairs:
        p_ids = tok.encode(p.prompt, add_special_tokens=True)[:-1]  # sin <eos> intermedio
        if len(p_ids) >= max_len - min_resp_len:
            # Prompt demasiado largo: respuesta no cabría en la ventana.
            # Se descarta el par — no es un hack, es un criterio de validez de DPO.
            n_discarded += 1
            continue
        for resp in (p.chosen, p.rejected):
            r_ids = tok.encode(resp, add_special_tokens=False) + [tok.eos_id]
            ids = (p_ids + r_ids)[:max_len]
            mask = [0] * len(p_ids) + [1] * len(r_ids)
            mask = mask[:max_len]
            pad = max_len - len(ids)
            seqs.append(ids + [tok.pad_id] * pad)
            resp_masks.append(mask + [0] * pad)
    if n_discarded:
        import warnings
        warnings.warn(f"collate_pairs: {n_discarded}/{len(pairs)} pares descartados "
                      f"(prompt >= max_len - {min_resp_len} = {max_len - min_resp_len} tokens). "
                      f"Considera aumentar max_len o filtrar el dataset.",
                      stacklevel=2)
    if not seqs:
        empty_ids = torch.zeros((0, max_len), dtype=torch.long, device=device)
        empty_mask = torch.zeros((0, max_len), dtype=torch.bool, device=device)
        return empty_ids, empty_mask, empty_mask
    input_ids = torch.tensor(seqs, dtype=torch.long, device=device)
    resp_mask = torch.tensor(resp_masks, dtype=torch.bool, device=device)
    attn = (input_ids != tok.pad_id)
    return input_ids, attn, resp_mask


def seq_logprobs(model, input_ids, attn, resp_mask):
    """Log-probabilidad TOTAL (suma) de los tokens de respuesta por secuencia. (B,)

    Implementa exactamente la ecuacion del paper DPO (Rafailov et al., 2023):
      log pi(y | x) = sum_t log pi(y_t | y_{<t}, x)

    Corrección numerica (no matematica):
    - logits.clamp(-1e4, 1e4): previene overflow de bfloat16 en MPS/CUDA que
      producen inf en log_softmax([inf, inf, ...]) = NaN. El rango [-1e4, 1e4]
      esta muy por encima de los logits tipicos de cualquier LM bien entrenado.
      Esta correccion NO cambia el gradiente en la region normal de operacion.
    """
    logits = model(input_ids=input_ids, attention_mask=attn).logits
    logits = logits.clamp(min=-1e4, max=1e4)  # guard numerica bfloat16/MPS
    logp = F.log_softmax(logits[:, :-1], dim=-1)
    tgt = input_ids[:, 1:]
    tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    mask = resp_mask[:, 1:] & (tgt != 0)
    # Suma cruda: semantica exacta de log pi(y|x) del paper DPO
    return (tok_logp * mask).sum(dim=-1)


def dpo_losses(model, ref_model, pairs, tok, max_len, beta, device,
               min_resp_len: int = 8):
    """Perdida DPO por par: vector (B_pairs,). Necesaria por separado para el
    probe de gradientes por ejemplo.

    Implementa la ecuacion 6 del paper DPO (Rafailov et al., 2023):
      l_i = -log sigma(beta * (Delta_theta_i - Delta_ref_i))
    donde Delta = log pi(y_w|x) - log pi(y_l|x), usando la SUMA de logprobs
    (no promedio), conforme al paper original.

    Guard de isfinite: pares descartados por collate_pairs (prompt demasiado largo)
    devuelven tensor vacio; si tras el filtrado el lote queda vacio, retorna tensor
    vacio para que el caller lo omita sin error.

    Device de ref_model: si ref_model esta en CPU y model en MPS, los tensores
    se mueven al device correcto para cada forward. El backward solo afecta a model.
    Esta separacion permite mantener ref_model en CPU para ahorrar RAM unificada.
    """
    input_ids, attn, resp_mask = collate_pairs(pairs, tok, max_len, device,
                                               min_resp_len=min_resp_len)
    if input_ids.shape[0] == 0:
        return torch.zeros((0,), device=device)
    lp = seq_logprobs(model, input_ids, attn, resp_mask)
    with torch.no_grad():
        # Detecta el device del modelo de referencia.
        # Si ref_model esta en CPU (ahorro de RAM), mueve tensores al CPU para el
        # forward de referencia y devuelve el resultado al device principal.
        ref_dev = next(ref_model.parameters()).device
        if ref_dev != input_ids.device:
            lr = seq_logprobs(ref_model,
                              input_ids.to(ref_dev),
                              attn.to(ref_dev),
                              resp_mask.to(ref_dev)).to(device)
        else:
            lr = seq_logprobs(ref_model, input_ids, attn, resp_mask)
    d_theta = lp[0::2] - lp[1::2]
    d_ref = lr[0::2] - lr[1::2]
    raw = -F.logsigmoid(beta * (d_theta - d_ref))
    # Guard: sustituye NaN/Inf residuales por log(2) (perdida neutra: modelo en empate)
    neutral = torch.full_like(raw, torch.log(torch.tensor(2.0)))
    return torch.where(torch.isfinite(raw), raw, neutral)


def train_dpo(model, ref_model, pairs, tok, cfg, device):
    """Entrena el checkpoint limpio con DPO estandar."""
    model.train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=cfg.lr)
    rng = torch.Generator().manual_seed(cfg.seed)
    n = len(pairs)
    bs = cfg.batch_size
    micro_bs = min(bs, getattr(cfg, "micro_batch_size", 16))
    for step in range(cfg.dpo_steps):
        idx = torch.randperm(n, generator=rng)[:bs].tolist()
        batch = [pairs[i] for i in idx]
        opt.zero_grad()
        for m_idx in range(0, len(batch), micro_bs):
            micro_batch = batch[m_idx:m_idx + micro_bs]
            losses = dpo_losses(model, ref_model, micro_batch, tok, cfg.max_len,
                                cfg.beta, device)
            if losses.shape[0] == 0:
                continue
            loss = (losses.mean() * len(micro_batch)) / len(batch)
            loss.backward()
        opt.step()
        if device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        if (step + 1) % max(1, cfg.dpo_steps // 5) == 0:
            print(f"  dpo step {step + 1}/{cfg.dpo_steps}  loss={loss.item():.4f}")
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    model.eval()
    return model
