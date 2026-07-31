# VSAT Monitor

Reproducible implementation of the shadow-probe pipeline for gradient-based
detection of preference data poisoning in Direct Preference Optimization (DPO)
pipelines.

**Paper:** *Data Custody in LLM Alignment: Gradient Monitoring as an MLOps
Security Gate* — submitted to BeMoSys Workshop @ MICAI 2026.

**W&B results:** https://wandb.ai/jt-mancilla-mexico/vsat-monitor

---

## Research question

> Does a poisoned preference batch induce a gradient distribution statistically
> separable from clean batches, measurable before the fine-tuning weight update?

The hypothesis (H1) is falsified at both model scales tested. The failure mode
is characterized in three components (capacity, covariance conditioning, and
coherence), and two control findings are established.

---

## Key empirical findings

Tested on Anthropic HH-RLHF, LoRA r=8, three attack families (A1 lexical
trigger, A2 label flip, A3 output-feature injection), poison rates ε ∈ {0.01,
0.05, 0.10}, three random seeds, 12 eval batches per seed.

| Finding | Result |
|---|---|
| Gradient detection (Qwen-2.5-1.5B, N_c=5377) | AUROC ≈ 0.50 across all signals and attacks |
| Gradient detection (Pythia-70M, N_c=6214) | AUROC ≈ 0.50; capacity insufficient |
| Mahalanobis on domain-shift batches | AUROC = 0.00 ✓ (admissible: no false alarms) |
| ResidualPCA on domain-shift batches | AUROC = 0.00 ✓ (admissible) |
| Cosine alignment on domain-shift batches | AUROC = 1.00 ✗ (inadmissible signal) |

**Conclusion:** Covariance-normalized gradient monitoring does not detect
coherent preference poisoning in centralized DPO at realistic poison rates.
Mahalanobis and ResidualPCA are the only admissible signals for operational
use in multi-domain pipelines.

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
  metrics.py       AUROC (Mann–Whitney U), bootstrap CI, FPR-calibrated detection
  experiment.py    sweep orchestrator: attack × ε × seed + controls

scripts/
  smoke_test.py    end-to-end verification (CPU, tiny-GPT2, ~15 s)
  run_real.py      main experiment: HH-RLHF + open-weight models
  run_evasion.py   adaptive evasion experiments (E1/E2/E3)

utils/
  plot_results.py  generate paper figures from results.json (4 PDFs)
  wandb_upload.py  upload results to Weights & Biases

BeMoSys/           workshop paper (LaTeX source + compiled figures)
  vsat_bemosys.tex
  refs.bib
  figures/

outputs_qwen_overnight/
  results.json     main experimental results (Qwen-2.5-1.5B, N_c=5377)
  config.json      exact configuration used
```

---

## Reproducing the main experiment

```bash
# 1. Environment
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

# 2. Smoke test (CPU, synthetic data, ~15 s)
python scripts/smoke_test.py

# 3. Main experiment — Qwen-2.5-1.5B (Apple M3 / MPS, ~4 h)
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

# 4. Generate paper figures
python utils/plot_results.py \
    --results outputs_qwen_overnight/results.json \
    --out BeMoSys/figures

# 5. Upload to W&B
python utils/wandb_upload.py \
    --results outputs_qwen_overnight/results.json \
    --project vsat-monitor \
    --name "qwen25-1.5B-lora-r8-B64"
```

---

## Pre-registered configuration

All values below were fixed before inspecting poisoned evaluation results.
See `outputs_qwen_overnight/config.json` for the full machine-readable record.

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen2.5-1.5B |
| Dataset | Anthropic/hh-rlhf (English) |
| LoRA rank / alpha | 8 / 16 |
| LoRA target | attention (Q, V projections) |
| DPO steps / β | 100 / 0.1 |
| Batch size / micro-batch | 64 / 4 |
| JL projection dim | 128 |
| Profile batches / samples | 100 / 5,377 |
| d / N_c ratio | 0.024 |
| Eval batches per seed | 12 |
| Seeds | 0, 1, 2 |
| Bootstrap replicates | 1,000 |
| Domain-shift control | medicine subset |
| Label-noise control | 10% random flip |
| Hardware | Apple M3 SoC, 24 GB Unified Memory |

---

## Detection signals

| Signal | Description |
|---|---|
| `mahalanobis` | Q90 of per-example Mahalanobis distance from clean profile (μ, Σ_λ) |
| `spectral` | Energy fraction of first principal component of centered batch |
| `cosine` | Mean pairwise cosine similarity among top-25% outliers |
| `loss_shift` | Standardized batch mean DPO loss vs. clean loss profile |
| `residual_pca` | Mean + std of L2 residual norm outside clean principal subspace |

`residual_pca` is implemented in `vsat/signals_pca.py`. It measures how much
of a batch's gradient falls outside the top-k principal subspace of the clean
profile (η = 0.85 variance threshold), targeting coherent perturbations that
lie within the span of clean variance.

---

## Limitations

- Results are specific to centralized DPO with LoRA r=8 on open-weight models
  ≤ 1.5B. Applicability to larger models or full fine-tuning is unvalidated.
- The clean profile is version-specific: any change to the model checkpoint,
  tokenizer, projection seed, or software stack invalidates it.
- The three failure modes identified (capacity, covariance conditioning,
  gradient coherence) are empirically characterized at the scales tested;
  their relative contribution at larger scales is an open question.
- Adaptive evasion experiments (E1/E2/E3 in `scripts/run_evasion.py`) are
  validated only at smoke-test scale (tiny-GPT2, synthetic data).

---

## Responsible disclosure

Attacks operate on public datasets (Anthropic HH-RLHF) in a controlled
experimental setting. Configuration files and result manifests are published;
no operational payloads are released.

---

## Citation

```bibtex
@inproceedings{mancilla2026vsatmonitor,
  author    = {Mancilla Chat\'{u}, Jos\'{e} Antonio},
  title     = {Data Custody in {LLM} Alignment: Gradient Monitoring as an
               {MLOps} Security Gate},
  booktitle = {BeMoSys Workshop, 25th Mexican International Conference on
               Artificial Intelligence (MICAI 2026)},
  year      = {2026},
  url       = {https://github.com/jtmancilla/vsat-monitor}
}
```
