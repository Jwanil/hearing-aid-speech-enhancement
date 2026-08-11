# Project Overview — Hearing Aid Speech Enhancement

> **Feed this file into every new AI session alongside `context.md` and `shared_context.md`.**
> This is the permanent project brief — it doesn't change. The logs change.

---

## One-Paragraph Summary (Memorize This)

We are building a deep learning speech denoiser for hearing aids that takes the user's individual audiogram — their personal hearing-loss profile — as a runtime input, using **FiLM conditioning** to adapt the model's noise-reduction behavior per person. Unlike existing generic denoisers that apply the same processing to everyone, our model produces different output for two different listeners hearing the same noisy speech. We evaluate using **HASPI and HASQI** — metrics specifically designed to predict intelligibility and quality for hearing-impaired listeners — and compare against both a classical Wiener filter and a non-personalized DNN baseline. We also report latency and model size, because real hearing aids need under 10ms delay, a constraint most academic research ignores.

---

## The Problem

- **466 million** people worldwide have disabling hearing loss (WHO)
- Hearing aids amplify sound but the #1 complaint is **understanding speech in noisy environments**
- Current noise reduction in hearing aids is **generic** — the same algorithm for every user
- But every person's hearing loss is different — a different audiogram, different frequencies affected, different severity
- Hearing aids already personalize **amplification** per audiogram — we're personalizing **noise reduction** too

---

## Our Technical Approach

### Architecture: U-Net + FiLM Conditioning

```
Noisy speech → STFT → Encoder (CNN) → FiLM Layer ← Audiogram vector
                                              ↓
                               Decoder → Ideal Ratio Mask
                                              ↓
                          Mask × Noisy spectrogram → ISTFT → Enhanced speech
```

### The Key Innovation: FiLM

**FiLM = Feature-wise Linear Modulation** (Perez et al., 2017)

The audiogram (6 hearing threshold values across frequencies) is fed into a small FiLM generator network, which outputs γ (scale) and β (shift) parameters. These modulate the internal feature maps of the U-Net:

```
output = γ × features + β
```

The model learns which internal detectors to amplify or suppress for each audiogram profile. Same architecture, different behavior per person.

### Three Models We Build and Compare

| Model | Description | Novelty |
|-------|-------------|---------|
| Wiener Filter | Classical DSP baseline | None (comparison point) |
| Generic DNN | U-Net with IRM masking, no personalization | Published approach |
| **Personalized DNN** | Same + FiLM conditioning on audiogram | **Our contribution** |

---

## Datasets

| Dataset | Use | Access |
|---------|-----|--------|
| **Clarity Challenge (CEC2/CEC3)** | Primary — includes audiograms, HA-specific | claritychallenge.org (free, register) |
| **VoiceBank-DEMAND** | Early experiments, standard benchmark | Available via SpeechBrain |

---

## Evaluation Metrics

| Metric | What it measures | Why we use it |
|--------|-----------------|---------------|
| **HASPI** | Intelligibility for hearing-impaired listeners | Core metric — accounts for audiogram |
| **HASQI** | Quality for hearing-impaired listeners | Complements HASPI |
| STOI/ESTOI | General intelligibility | Standard comparison |
| SI-SDR | Signal quality in dB | Standard comparison |
| Latency (ms) | Processing delay | Real-world constraint |
| Parameter count | Model size | Real-world constraint |

Tool: `pyclarity` (free, open-source) computes HASPI/HASQI.

---

## Key Constraint: Latency

Real hearing aids need **<10ms processing delay** to avoid lip-sync mismatch. Most academic models ignore this. We report it honestly and discuss what would be needed for real deployment (quantization, pruning — future work).

---

## Novelty Claim

Generic DNN denoising exists (Audatic, Nature Scientific Reports 2023 — restored near-normal intelligibility). The specific combination of:
1. Audiogram-conditioned noise reduction at **runtime** (not just at training time)
2. Within a **latency-aware, small-footprint** design
3. Evaluated with **hearing-loss-specific metrics** (HASPI/HASQI)

...is the underexplored gap we're targeting.

---

## Tech Stack

```
Core:        PyTorch + torchaudio
Baselines:   SpeechBrain, Asteroid
Evaluation:  pyclarity (HASPI/HASQI), pystoi (STOI)
Notebooks:   Jupyter
Version:     Git + GitHub
```

---

## Project Phases

| Phase | Name | Directive |
|-------|------|-----------|
| 0 | Audio/DSP Fundamentals | `directives/00_dsp_fundamentals.md` |
| 1 | Audiology + Audiograms | `directives/01_audiology.md` |
| 2 | Wiener Filter Baseline | `directives/02_wiener_baseline.md` |
| 3 | Data Pipeline | `directives/03_data_pipeline.md` |
| 4 | Generic DNN Denoiser | `directives/04_generic_dnn.md` |
| 5 | FiLM Conditioning | `directives/05_film_conditioning.md` |
| 6 | Full Evaluation | `directives/06_evaluation.md` |
| 7 | Report + Demo | `directives/07_report.md` |

---

## Key Papers

1. **Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer" (2017)** — the core technique. Short, readable.
2. **Diehl et al., Nature Scientific Reports (2023)** — strongest existing work, what we're differentiating from.
3. **NeuroAMP (arXiv 2025)** — closest to our personalization angle (but for amplification, not denoising).
4. **Luo & Mesgarani, Conv-TasNet (2019)** — architecture reference.
5. **Clarity Challenge overview papers** — benchmark context.

---

## Team

- **Jwanil** — Lead developer, project ideation
- **[Partner Name]** — Co-developer

**Workflow:** VS Code Live Share for real-time sessions + Git/GitHub for async sync.

---

## For Agent Sessions

When starting a new session with this file as context:
1. Read `context.md` for current task state
2. Read `shared_context.md` for what your partner has done
3. Check the phase status table in `context.md`
4. Begin where the log says to begin
5. After every action, update `context.md` and `shared_context.md`
