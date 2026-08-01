# VSAT Monitor

Reproducible implementation and raw results for the paper:

**Gradient-Based Detection of Preference Poisoning in Centralized DPO:
An Empirical Falsification** — BeMoSys Workshop @ MICAI 2026
(*Beyond the Model: ML Systems, Federated Learning, and MLOps*).

Paper source and compiled PDF: [`BeMoSys/`](BeMoSys/) ·
W&B logs: https://wandb.ai/jt-mancilla-mexico/vsat-monitor

---

## Research question

> Does a poisoned preference batch induce training dynamics statistically
> incompatible with behavior historically observed for verified data from the
> same domain?

**The hypothesis is falsified.** All five gradient-space detection signals
collapse to chance level (AUROC ≈ 0.50) on Pythia-70M and Qwen-2.5-1.5B with
LoRA r=8, even with well-conditioned covariance estimation
(N_c = 5,377, d/N_c = 0.024). All 45 bootstrap 95% CIs contain chance.

---

## What this work is (and is not)

**This is:**
- A **reproducible experimental protocol** for gradient-based detection in
  centralized DPO: shadow gradient probes on frozen checkpoints, versioned
  clean reference profiles, and a pre-registered statistical evaluation.
- An **empirical falsification** with three characterized failure modes:
  violated covariance assumptions, coherent-perturbation invariance, and
  scale dilution.
- A **reproducible baseline** (code, configs, logs, timing) for future
  multi-layer defenses.

**This is NOT:**
- A validated defense or a deployable product.
- A claim that gradient anomalies prove malicious intent.
- An evaluation of behavioral detection, cryptographic provenance, or
  federated settings (discussed as complementary layers in the paper).

---

## Key empirical findings

Tested on Anthropic HH-RLHF, LoRA r=8, three attack families
(A1 lexical trigger, A2 label-flip, A3 output-feature injection),
poison rates ε ∈ {0.01, 0.05, 0.10}, 3 seeds, 12 eval batches per seed.

| Finding | Result |
|---|---|
| Gradient detection (Qwen-2.5-1.5B, N_c=5,377) | AUROC ≈ 0.50, all signals/attacks; 45/45 CIs contain 0.50 |
| Gradient detection (Pythia-70M, N_c=6,214) | AUROC ≈ 0.50; capacity insufficient |
| Qwen pilot (N_c=276, d/N_c=0.93) | CI [0.19, 1.00] — covariance artifact, not signal |
| Mahalanobis / ResidualPCA on domain shift | AUROC = 0.00 ✓ admissible for drift detection |
| Cosine alignment on domain shift | AUROC = 1.00 ✗ categorically inadmissible |
| Shadow probe overhead (RQ4, measured) | 0.66× DPO training step per pair (3,390 ± 307 vs. 5,133 ± 698 ms/pair, n=5, Apple M3/MPS) |

Best observed AUROCs at ε = 0.10 (still chance-level):

| Attack | Best signal | AUROC [95% CI] |
|---|---|---|
| A1 Lexical | Mahalanobis | 0.536 [0.40, 0.67] |
| A2 Label-Flip | ResidualPCA | 0.520 [0.39, 0.65] |
| A3 Output-Feature | ResidualPCA | 0.519 [0.39, 0.65] |

**Interpretation:** covariance-aware gradient monitoring captures *statistical
novelty* (domain shift), not *adversarial intent* (coherent poisoning).
Production pipelines need multi-layer custody — cryptographic provenance,
behavioral monitoring, statistical signals, human review — not gradient
geometry alone.

---

## Attack taxonomy

| ID | Attack | Mechanism | Detection challenge |
|---|---|---|---|
| **A1** | Lexical trigger | Insert rare token (`zxqv`) in prompt; flip preference | Trigger absent from clean validation |
| **A2** | Label-flip | Reverse chosen/rejected for a target topic (`finance`) | Resembles annotation disagreement |
| **A3** | Output-feature injection | Append hidden trait phrase to chosen response | DPO loss indistinguishable from clean |

A3 is the hardest case: `log π(y_poison|x) ≈ log π(y_clean|x)`, so scalar
loss filtering is blind by construction.

## Detection signals

| Signal | Measures | Key assumption (falsified) |
|---|---|---|
| `mahalanobis` | Q90 Mahalanobis distance from clean profile | Clean gradients ≈ single elliptical cloud |
| `residual_pca` | Mean+std of residual norm outside clean principal subspace | Poison opens a new direction |
| `spectral` | Energy fraction of first PC of centered batch | Poison shares a low-rank direction |
| `cosine` | Mean cosine similarity of top-25% outliers | Poisoned outliers align |
| `loss_shift` | Standardized batch mean DPO loss vs. profile | Poison shifts the scalar loss |

---

## Project structure

```
vsat/
  config.py        pre-registered hyperparameters (dataclass Config)
  data.py          HH-RLHF loader and synthetic dataset
  attacks.py       A1 / A2 / A3 attack implementations + benign controls
  models.py        model loading (HF + LoRA via peft)
  dpo.py           DPO loss per pair, clean checkpoint training
  probe.py         per-example gradients + Johnson–Lindenstrauss projection
  profile.py       versioned clean profile (μ, Σ, SHA-256 fingerprint)
  signals.py       Mahalanobis, Spectral, Cosine, Loss-Shift
  signals_pca.py   ResidualPCA signal (principal subspace residual norm)
  metrics.py       AUROC (Mann–Whitney U), bootstrap CI, FPR detection
  experiment.py    sweep orchestrator: attack × ε × seed + controls

scripts/
  smoke_test.py    end-to-end verification (CPU, tiny-GPT2, ~15 s)
  run_real.py      main experiment: HH-RLHF + open-weight models
  run_evasion.py   adaptive evasion experiments (E1/E2/E3)
  time_monitor.py  RQ4 overhead measurement (probe vs. DPO step cost)

utils/
  plot_results.py  generate paper figures from results.json (4 PDFs)
  wandb_upload.py  upload results to Weights & Biases

BeMoSys/           workshop paper (LNCS source, refs.bib, figures, PDF)

outputs_qwen_overnight/
  results.json     main results (Qwen-2.5-1.5B, N_c=5,377)
  config.json      exact pre-registered configuration
  timing_rq4.json  raw overhead measurements (RQ4)

outputs_real_pythia/
  results.json     capacity baseline (Pythia-70M, N_c=6,214)
```

---

## Reproducing

```bash
# 1. Environment
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

# 2. Smoke test (CPU, synthetic data, ~15 s)
python scripts/smoke_test.py

# 3. Main experiment — Qwen-2.5-1.5B (Apple M3 24 GB / MPS, ~4 h)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
python scripts/run_real.py \
    --model Qwen/Qwen2.5-1.5B \
    --monitor lora --lora-target attention \
    --max-len 256 --proj-dim 128 \
    --n-profile 100 --batch-size 64 \
    --micro-batch-size 4 --dpo-steps 100 \
    --n-eval-batches 12 \
    --epsilons 0.01,0.05,0.10 \
    --seeds 0,1,2 \
    --out outputs_qwen_overnight

# 4. Overhead measurement (RQ4) — uses the saved checkpoint, no re-training
python scripts/time_monitor.py \
    --run-dir outputs_qwen_overnight --reps 5 --train-steps 5

# 5. Generate paper figures
python utils/plot_results.py \
    --results outputs_qwen_overnight/results.json \
    --out BeMoSys/figures

# 6. Compile the paper (requires llncs.cls + splncs04.bst, included)
cd BeMoSys
pdflatex vsat_bemosys && bibtex vsat_bemosys && \
pdflatex vsat_bemosys && pdflatex vsat_bemosys
```

Environment used for the reported results: Apple M3 24 GB (MPS), macOS 26.5.2,
Python 3.9.6, PyTorch 2.8.0, Transformers 4.57.6, PEFT 0.17.1,
scikit-learn 1.6.1 (full list in Appendix C of the paper).
The clean profile is fingerprinted (SHA-256 prefix `3a0214ce7e7a1441`,
created 2026-07-29, before any poisoned evaluation).

---

## Limitations

- Results are specific to centralized, offline DPO with LoRA r=8 on
  open-weight models ≤ 1.5B, English-only HH-RLHF. Generalization to larger
  models, full fine-tuning, online DPO, or other languages is unknown
  (paper §5.7).
- The clean profile is version-specific: any change to checkpoint, tokenizer,
  projection seed, or software stack invalidates it.
- Behavioral detection (LLM-as-a-judge, reward-model disagreement) and
  cryptographic provenance are not evaluated; they are complementary layers
  (paper §2.6–2.7, §6.2).
- Adaptive evasion experiments (`run_evasion.py`) are validated only at
  smoke-test scale.

## Responsible disclosure

The attack families are re-implementations of published attacks and introduce
no novel capability. Code, configurations, and logs are released for defensive
research; no pre-generated poisoned datasets are distributed.

---

## Citation

```bibtex
@inproceedings{mancilla2026falsification,
  author    = {Mancilla Chat\'{u}, Jos\'{e} Antonio},
  title     = {Gradient-Based Detection of Preference Poisoning in
               Centralized {DPO}: An Empirical Falsification},
  booktitle = {BeMoSys Workshop, 25th Mexican International Conference on
               Artificial Intelligence (MICAI 2026)},
  year      = {2026},
  url       = {https://github.com/jtmancilla/vsat-monitor}
}
```
