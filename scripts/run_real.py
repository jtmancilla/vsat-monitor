"""Ejecucion real (Fase 1) con modelo HF + HH-RLHF.

Pensado para Mac M3 / GPU con Qwen2.5-1.5B / Llama-3.2-1B + LoRA.
Replica la configuracion del diseno experimental, con parametros calibrables
por CLI para ajustar el costo segun el hardware disponible.

Uso tipico (Mac M3 overnight, ~6h):
  python scripts/run_real.py \\
      --model Qwen/Qwen2.5-1.5B \\
      --proj-dim 256 --max-len 256 --n-profile 50 --n-eval-batches 4 \\
      --seeds 0 --epsilons 0.01,0.10 --out outputs_real_qwen

Uso formal completo (GPU, ~12h):
  python scripts/run_real.py --model Qwen/Qwen2.5-1.5B --out outputs_real_qwen
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from vsat.config import Config
from vsat.data import load_pairs
from vsat.dpo import train_dpo
from vsat.experiment import run_experiment, _SIGNALS
from vsat.models import build_model_and_tokenizer


def main():
    import sys
    # Forzar flush por linea aunque stdout vaya a un pipe (| tee).
    # Sin esto, los print() se bloquean en buffer de 4KB y no aparecen hasta
    # que el buffer esta lleno o el proceso termina.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass  # Python < 3.7

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--out", default="outputs_real")
    ap.add_argument("--n-eval-batches", type=int, default=0,
                    help="Lotes de evaluacion por (ataque,eps,semilla). 0 = auto (12 formal, 4 fast).")
    ap.add_argument("--proj-dim", type=int, default=128,
                    help="Dimension de la proyeccion JL (128 para Qwen en Mac M3).")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--micro-batch-size", type=int, default=16,
                    help="Micro-lote para GPU/MPS (evita OOM en logits).")
    ap.add_argument("--dpo-steps", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=256,
                    help="Longitud maxima de secuencia (256 recomendado para Qwen en Mac M3).")
    ap.add_argument("--n-profile", "--n-profile-batches", type=int, default=0, dest="n_profile",
                    help="Lotes para estimar el perfil limpio. 0 = auto (100 formal, 20 fast).")
    ap.add_argument("--seeds", default="",
                    help="Semillas separadas por coma (ej. '0,1,2'). Vacio = auto.")
    ap.add_argument("--epsilons", default="",
                    help="Epsilons separados por coma (ej. '0.01,0.05,0.10'). Vacio = auto.")
    ap.add_argument("--monitor", default="lora",
                    help="Parametros a monitorear: lora, all, last_layer_head, etc.")
    ap.add_argument("--lora-target", default="attention",
                    help="Target modules para LoRA: attention, all, etc.")
    ap.add_argument("--fast", action="store_true",
                    help="Modo rapido para verificacion.")
    args = ap.parse_args()

    # --- Resolver valores auto -----------------------------------------------
    seeds_auto = tuple(int(s) for s in args.seeds.split(",")) \
        if args.seeds else None
    epsilons_auto = tuple(float(e) for e in args.epsilons.split(",")) \
        if args.epsilons else None

    if args.fast:
        cfg = Config(
            dataset="hh-rlhf" if args.model != "tiny-gpt2" else "synthetic",
            model=args.model,
            batch_size=16,
            micro_batch_size=args.micro_batch_size,
            dpo_steps=30,
            proj_dim=min(args.proj_dim, 128),
            n_clean_train=160,
            n_profile_batches=args.n_profile or 20,
            epsilons=epsilons_auto or (0.05, 0.10),
            seeds=seeds_auto or (0,),
            max_len=args.max_len or 256,
            min_resp_len=8,
            lr=5e-5,
            out_dir=args.out,
            monitor=args.monitor,
            lora_target_modules=args.lora_target,
        )
        n_eval = args.n_eval_batches or 4
    else:
        cfg = Config(
            dataset="hh-rlhf" if args.model != "tiny-gpt2" else "synthetic",
            model=args.model,
            batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
            dpo_steps=args.dpo_steps,
            proj_dim=args.proj_dim,
            n_profile_batches=args.n_profile or 100,
            epsilons=epsilons_auto or (0.01, 0.05, 0.10),
            seeds=seeds_auto or (0, 1, 2),
            max_len=args.max_len or 256,
            min_resp_len=8,
            lr=5e-5,
            out_dir=args.out,
            monitor=args.monitor,
            lora_target_modules=args.lora_target,
        )
        n_eval = args.n_eval_batches or 12

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # --- Estimacion de tiempo ------------------------------------------------
    # ~7.5 seg/par para Qwen-1.5B en MPS M3 (medido en fast mode); ~3 seg en CPU
    n_probe_pairs = (cfg.n_profile_batches * cfg.batch_size
                     + len(cfg.epsilons) * len(cfg.seeds)
                       * n_eval * 3 * cfg.batch_size * 2)  # 3 ataques, pos+neg
    secs_per_pair = 0.05 if device == "mps" else 3.0
    est_h = (cfg.dpo_steps * 0.2 + n_probe_pairs * secs_per_pair) / 3600
    print(f"device={device}  model={cfg.model}  "
          f"max_len={cfg.max_len}  proj_dim={cfg.proj_dim}", flush=True)
    print(f"  n_profile={cfg.n_profile_batches}x{cfg.batch_size}  "
          f"n_eval={n_eval}  seeds={cfg.seeds}  eps={cfg.epsilons}", flush=True)
    print(f"  estimado: {est_h:.2f} h ({est_h*60:.1f} min)  "
          f"({n_probe_pairs:,} pares a probar + {cfg.dpo_steps} pasos DPO)", flush=True)

    pairs = load_pairs(cfg)
    model, ref_model, tok = build_model_and_tokenizer(cfg)
    model.to(device)
    ref_model.to(device)
    ref_model_device = str(next(ref_model.parameters()).device)
    print(f"  model -> {device}  |  ref_model -> {ref_model_device} (100% aceleracion Metal GPU)", flush=True)
    os.makedirs(cfg.out_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.out_dir, "config.json"))

    print("[setup] entrenando checkpoint DPO limpio ...")
    model = train_dpo(model, ref_model, pairs[:cfg.n_clean_train], tok, cfg, device)

    print(f"[setup] guardando adaptador LoRA en {cfg.out_dir}/dpo_checkpoint ...")
    model.save_pretrained(os.path.join(cfg.out_dir, "dpo_checkpoint"))

    print("[experimento] barrido completo ...")
    out = run_experiment(model, ref_model, tok, pairs, cfg, device,
                         n_eval_batches=n_eval)

    print("\n=== AUROC por ataque y senal ===")
    first_row = next(iter(out["results"].values()))
    signals_print = [s for s in _SIGNALS if s in first_row]
    for key, row in out["results"].items():
        line = f"{key:32s}"
        for s in signals_print:
            line += f"  {s[:6]}={row[s]['auroc']:.3f}"
        print(line)
    print("\n=== Controles ===")
    for s, c in out["controls"].items():
        print(f"{s:16s} shift={c['shift_vs_clean_auroc']:.2f} "
              f"noise={c['noise_vs_clean_auroc']:.2f}")


if __name__ == "__main__":
    main()
