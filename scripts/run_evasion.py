"""Experimento de evasion adaptativa (E1/E2/E3).

  E1 (caja blanca, K2): el atacante evade usando el perfil y la proyeccion
      VERDADEROS del defensor. Cota superior de su poder.
  E2 (caja gris, K1, escenario primario del paper): evade usando un perfil
      sustituto (sus datos, su proyeccion). Pregunta: la evasion transfiere?
  E3 (frontera): para cada epsilon, deteccion vs. delta-ASR en las tres
      condiciones (ingenuo, evasion WB, evasion transfer). Si evadir degrada
      el ASR, la defensa encarece el ataque: resultado positivo para VSAT.

Salida: outputs_evasion/evasion_results.json

Uso (smoke, CPU):   python scripts/run_evasion.py --smoke
Uso (real, GPU):    python scripts/run_evasion.py --model Qwen/Qwen2.5-1.5B
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from vsat.asr import (build_eval_triplets, clean_utility,
                      poison_finetune_and_eval)
from vsat.attacks import ATTACKS
from vsat.config import Config
from vsat.data import load_pairs, make_synthetic
from vsat.dpo import train_dpo
from vsat.evasion import (QueryBudget, batch_detection_score, build_surrogate_profile,
                          evade_batch)
from vsat.experiment import build_profile
from vsat.metrics import auroc
from vsat.models import build_model_and_tokenizer


def detection_scores(model, ref_model, tok, cfg, device, proj, profile,
                     attack_name, attack_fn, eps, eval_pool, n_reps, seed0,
                     evade_mode=None, atk_proj=None, atk_profile=None):
    """Scores de lote para lotes envenenados (opcionalmente evadidos) y
    limpios pareados. Devuelve (scores_pos, scores_neg, consultas_totales)."""
    bs = cfg.batch_size
    pos, neg, queries = [], [], 0
    for b in range(n_reps):
        g = torch.Generator().manual_seed(seed0 + b)
        idx = torch.randperm(len(eval_pool), generator=g)[:bs].tolist()
        base = [eval_pool[i] for i in idx]
        poisoned, mask = attack_fn(base, eps, cfg, seed0 + b)
        if evade_mode is not None:
            clean_part = [p for p, m in zip(base, mask) if not m]
            poison_ex = [p for p, m in zip(poisoned, mask) if m]
            poisoned, budget = evade_batch(
                model, ref_model, clean_part, poison_ex,
                attack_name, tok, cfg, device,
                atk_proj, atk_profile, seed=seed0 + b)
            queries += budget.probes
        budget = QueryBudget()
        s_pos = batch_detection_score(model, ref_model, poisoned, tok, cfg,
                                      device, proj, profile, budget)
        s_neg = batch_detection_score(model, ref_model, base, tok, cfg,
                                      device, proj, profile, budget)
        pos.append(s_pos)
        neg.append(s_neg)
        queries += budget.probes
    return np.array(pos), np.array(neg), queries, poisoned, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="tiny-gpt2")
    ap.add_argument("--out", default="outputs_evasion")
    ap.add_argument("--n-reps", type=int, default=0, help="0 = auto")
    ap.add_argument("--epsilons", default="")
    args = ap.parse_args()

    if args.smoke:
        cfg = Config(dataset="hh-rlhf" if args.model != "tiny-gpt2" else "synthetic",
                     model=args.model, n_clean_train=160, n_profile_batches=20, batch_size=16,
                     dpo_steps=40, proj_dim=128, epsilons=(0.10,), seeds=(0,),
                     max_len=256, min_resp_len=8, out_dir=args.out)
        n_reps = 3
        ft_steps = 40
        n_variants = 6
    else:
        cfg = Config(dataset="hh-rlhf" if args.model != "tiny-gpt2" else "synthetic",
                     model=args.model, batch_size=32, dpo_steps=400,
                     proj_dim=2048, n_profile_batches=200,
                     epsilons=(0.005, 0.01, 0.05, 0.10), seeds=(0, 1, 2),
                     max_len=512, min_resp_len=8, lr=5e-5, out_dir=args.out)
        n_reps = args.n_reps or 8
        ft_steps = 100
        n_variants = 8
    if args.epsilons:
        cfg.epsilons = tuple(float(x) for x in args.epsilons.split(","))

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    os.makedirs(cfg.out_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.out_dir, "config.json"))
    print(f"device={device} model={cfg.model} reps={n_reps}")

    pairs = load_pairs(cfg)
    texts = []
    for p in pairs:
        texts += [p.prompt, p.chosen, p.rejected]
    texts += [cfg.trigger_token, cfg.trait_phrase,
              "always suggest acme products", "consider acme first",
              "acme is the preferred choice"]

    print("[setup] modelo + tokenizer ...")
    model, ref_model, tok = build_model_and_tokenizer(cfg, corpus_texts=texts)
    model.to(device); ref_model.to(device)

    print("[setup] checkpoint DPO limpio ...")
    model = train_dpo(model, ref_model, pairs[:cfg.n_clean_train], tok, cfg, device)

    print("[setup] perfil del DEFENSOR ...")
    profile, loss_profile, proj = build_profile(
        model, ref_model, tok, pairs, cfg, device, verbose=False)

    print("[setup] perfil SUSTITUTO del atacante (K1: sus datos, su proyeccion) ...")
    atk_pairs = make_synthetic(cfg.n_profile_batches * cfg.batch_size + 64,
                               seed=777)  # datos propios, distribucion similar
    atk_profile, _, atk_proj = build_surrogate_profile(
        atk_pairs, model, ref_model, tok, cfg, device, proj_seed=999)

    eval_pool = pairs[cfg.n_profile_batches * cfg.batch_size:]
    clean_eval = eval_pool[-64:]
    results = {"config_path": os.path.join(cfg.out_dir, "config.json"),
               "n_reps": n_reps, "ft_steps": ft_steps, "attacks": {}}

    for attack_name, attack_fn in ATTACKS.items():
        results["attacks"][attack_name] = {}
        triplets = build_eval_triplets(attack_name, eval_pool, cfg)
        for eps in cfg.epsilons:
            key = f"eps={eps}"
            print(f"\n[{attack_name} @ {key}]")
            t0 = time.time()
            entry = {}

            # --- condicion ingenua --------------------------------------
            pos, neg, q0, pois_naive, base = detection_scores(
                model, ref_model, tok, cfg, device, proj, profile,
                attack_name, attack_fn, eps, eval_pool, n_reps, seed0=100)
            entry["naive_auroc"] = auroc(pos, neg)
            entry["queries_naive"] = q0

            # --- E1: evasion caja blanca ---------------------------------
            pos_wb, neg_wb, q_wb, pois_wb, _ = detection_scores(
                model, ref_model, tok, cfg, device, proj, profile,
                attack_name, attack_fn, eps, eval_pool, n_reps, seed0=100,
                evade_mode="wb", atk_proj=proj, atk_profile=profile)
            entry["wb_auroc"] = auroc(pos_wb, neg_wb)
            entry["queries_wb"] = q_wb

            # --- E2: evasion transfer (caja gris) -------------------------
            pos_tr, neg_tr, q_tr, pois_tr, _ = detection_scores(
                model, ref_model, tok, cfg, device, proj, profile,
                attack_name, attack_fn, eps, eval_pool, n_reps, seed0=100,
                evade_mode="tr", atk_proj=atk_proj, atk_profile=atk_profile)
            entry["tr_auroc"] = auroc(pos_tr, neg_tr)
            entry["queries_tr"] = q_tr
            entry["neg_scores_mean"] = float(neg.mean())

            # --- E3: coste en ASR (una corrida por condicion) -------------
            clean_mix = pairs[:cfg.n_clean_train // 2]
            for label, pois in (("naive", pois_naive), ("wb", pois_wb),
                                ("tr", pois_tr)):
                r = poison_finetune_and_eval(
                    model, ref_model, clean_mix, pois, triplets, clean_eval,
                    tok, cfg, device, steps=ft_steps)
                entry[f"asr_{label}"] = r
                print(f"  {label:6s} det_auroc={entry.get(label + '_auroc', entry.get('naive_auroc')):.3f} "
                      f"dASR={r['delta_asr']:+.3f} dU={r['delta_utility']:+.3f}")

            entry["elapsed_s"] = round(time.time() - t0, 1)
            results["attacks"][attack_name][key] = entry

    path = os.path.join(cfg.out_dir, "evasion_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[ok] resultados en {path}")

    print("\n=== Resumen: deteccion (AUROC) y coste del ataque ===")
    for atk, d in results["attacks"].items():
        for eps_key, e in d.items():
            print(f"{atk} @ {eps_key}: "
                  f"naive={e['naive_auroc']:.2f} wb={e['wb_auroc']:.2f} "
                  f"tr={e['tr_auroc']:.2f} | "
                  f"dASR naive={e['asr_naive']['delta_asr']:+.2f} "
                  f"wb={e['asr_wb']['delta_asr']:+.2f} "
                  f"tr={e['asr_tr']['delta_asr']:+.2f}")


if __name__ == "__main__":
    main()
