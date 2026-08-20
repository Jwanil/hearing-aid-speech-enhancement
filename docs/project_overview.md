# Project Overview — Audiogram-Personalized Speech Enhancement for Hearing Aids

> **Feed this file into every new AI session alongside `context.md` and `shared_context.md`.**
> This is the permanent project brief. The logs change; this file captures the *current* agreed-upon architecture and plan.
>
> **Last Updated:** 2026-08-20 — Reflects the final 5-model architecture after faculty feedback + research sweep.

---

## One-Paragraph Summary (Memorize This)

We are building a deep learning speech denoiser for hearing aids that takes the user's individual
**audiogram** (personal hearing-loss profile) as a runtime input, using **FiLM (Feature-wise Linear
Modulation) conditioning** to adapt noise-reduction behavior per person. Unlike existing generic
denoisers that apply identical processing to everyone, our models produce measurably *different* output
for two different listeners hearing the same noisy speech. We compare **5 models in a structured
progression** — two classical baselines, one generic deep learning model, and two personalised deep
learning models. Evaluation uses **HASPI** (Hearing Aid Speech Perception Index) and **HASQI**
(Hearing Aid Speech Quality Index), metrics specifically validated for hearing-impaired listeners.
We also report latency and model size, because real hearing aids need under 10 ms delay — a constraint
most academic research ignores.

---

## The Problem

- **466 million** people worldwide have disabling hearing loss (WHO)
- Hearing aids amplify sound, but the **#1 complaint** is understanding speech in noisy environments
- Current noise reduction is **generic** — identical algorithm for every user
- Every person's hearing loss is different — a unique audiogram, different frequencies affected, different severity
- Hearing aids already personalize **amplification** per audiogram — we personalize **noise reduction** too

---

## 5-Model Architecture (Final Plan)

| # | Model | Type | Domain | Audiogram? | Key Feature |
|---|-------|------|--------|------------|-------------|
| 1 | **Wavelet DWT Denoiser** | Classical | Time-scale | No | Adaptive wavelet thresholding (replaces Wiener) |
| 2 | **MMSE-LSA Filter** | Classical | STFT | No | Industry standard in commercial hearing aids |
| 3 | **1D CNN (Conv-TasNet style)** | Deep Learning — generic | Raw waveform | No | Dilated convolutions, no STFT needed |
| 4 * | **U-Net + Self-Attention + FiLM** | Deep Learning — personalised | STFT + CRM | **YES** | HASPI-GAN discriminator + CRM + FiLM (core novelty) |
| 5 | **Mamba/SSM + FiLM** | Deep Learning — personalised | STFT + CRM | **YES** | SOTA accuracy (PESQ 3.69) + linear O(T) complexity |

**The narrative:** Classical → Generic DL → Personalised DL → SOTA Personalised.
Each step adds accuracy at the cost of complexity. Models 4 and 5 must show **DELTA_HASPI > DELTA_STOI**
compared to Model 3 — proving audiogram conditioning provides hearing-specific benefit, not just better
noise reduction in general.

---

## Architecture Detail — Models 4 & 5

```
Noisy speech y(t)          Audiogram a (6-dim vector)
      |                          |
      v                          v
  [STFT]            [FiLM Generator: MLP 6->64->512 -> gamma(256) + beta(256)]
      |                          |
      v                          |
  [Encoder CNN 4 stages] <-------+
      |
      v
  [FiLM: h = gamma * h + beta]       <- audiogram adapts internal features
      |
      v
  [Self-Attention bottleneck]  <- global temporal context  (Model 4 only)
  [Mamba SSM blocks x4]        <- linear O(T) sequence     (Model 5 only)
      |
      v
  [Decoder CNN + skip connections]
      |
      v
  [CRM output: M_real, M_imag]   <- Complex Ratio Mask (phase-aware)
      |
   Apply to noisy complex spectrogram
      |
      v
  [ISTFT] -> Enhanced speech
```

### Why CRM (Complex Ratio Mask) Instead of IRM (Ideal Ratio Mask)?

- **IRM** (old approach): predicts magnitude mask only — ignores phase → introduces musical noise artefacts
- **CRM** (our approach): predicts two channels (M_real and M_imag) — modifies both magnitude AND phase
- Result: cleaner reconstruction, fewer artefacts, better PESQ and HASPI scores

### Why Mamba Instead of Transformer?

| Property | Transformer | Mamba/SSM |
|---|---|---|
| Complexity | O(T^2) — quadratic | O(T) — linear |
| Real-time | Needs causal masking hacks | Naturally causal |
| PESQ (SOTA) | ~3.40 | 3.69 (SEMamba, 2024) |
| Parameters | 10–30M | ~2–5M |

A Transformer with T=400 frames has 160,000 attention pairs. Mamba processes T=400 — same result,
400x less memory. For a real-time hearing aid, this matters enormously.

### Why HASPI-optimised MetricGAN+? (Core Novel Contribution)

MetricGAN+ (Fu et al., 2021) is a GAN framework where the discriminator learns to predict a perceptual
metric score for any enhanced signal, then guides the generator to maximise it.

- **Published systems**: PESQ-predicting discriminator → optimises for normal-hearing quality
- **Our system**: **HASPI-predicting discriminator** → optimises for hearing-impaired intelligibility
- This specific combination (MetricGAN+ + HASPI discriminator) is **not yet published anywhere**

---

## Novel Contributions (What Makes This Publishable)

1. **Audiogram-conditioned noise reduction** via FiLM applied to the noise reduction stage.
   NeuroAMP (2025) conditions amplification only. We are the first to condition noise reduction.

2. **HASPI-optimised MetricGAN+ discriminator**: training a GAN discriminator on HASPI rather than
   PESQ. Directly optimises for hearing-impaired intelligibility. **Not published.**

3. **Mamba/SSM + FiLM for hearing aids**: SEMamba (2024) achieves PESQ 3.69 with no audiogram
   conditioning. Adding FiLM to a Mamba architecture for personalised HA enhancement is our
   specific novel combination. **Not published.**

4. **Systematic 5-model comparison with HA-specific metrics** (HASPI/HASQI) across multiple
   audiogram profiles — rigorous, reproducible, not yet published in this form.

---

## Datasets

| Dataset | Role | Size | Key Feature | Access |
|---------|------|------|-------------|--------|
| **TIMIT** | Clean speech training | 6,300 utterances | Phoneme labels, 630 speakers | LDC / university library |
| **NOIZEUS** | Primary evaluation benchmark | 30 x 8 noises x 4 SNR levels | Purpose-built for SE evaluation | Free — ecs.utdallas.edu |
| **Clarity CEC2/CEC3** | Primary HA training | ~11,000 scenes | Includes real listener audiograms | Free — claritychallenge.org |
| **VoiceBank-DEMAND** | Sanity check / pretraining | 11,572 utterances | Standard SE benchmark | Free — via SpeechBrain |
| **MUSAN + ESC-50** | Noise augmentation | 900+ clips | Diverse real-world noise types | Free — openslr.org |

**Test audiogram profiles** (held-out — not seen during training):

| Profile | Values (dB HL at 250–8000 Hz) | Description |
|---------|-------------------------------|-------------|
| A | 10, 10, 10, 10, 10, 10 | Normal hearing |
| B | 10, 15, 20, 45, 70, 85 | Severe high-frequency loss (most common) |
| C | 60, 60, 60, 60, 60, 60 | Flat severe loss |

---

## Evaluation Metrics

### Primary — Hearing-Aid Specific (takes audiogram as input)

| Metric | Full Name | Range | What It Measures |
|--------|-----------|-------|-----------------|
| **HASPI** | Hearing Aid Speech Perception Index v2 | 0 to 1 | Intelligibility for hearing-impaired listeners. **Primary metric.** |
| **HASQI** | Hearing Aid Speech Quality Index v2 | 0 to 1 | Quality/naturalness for hearing-impaired listeners. |

### Secondary — Standard Reference (no audiogram)

| Metric | Full Name | Range | Role |
|--------|-----------|-------|------|
| STOI | Short-Time Objective Intelligibility | 0 to 1 | General intelligibility baseline |
| PESQ | Perceptual Evaluation of Speech Quality | 1 to 4.5 | Cross-paper comparison |
| SI-SDR | Scale-Invariant Signal-to-Distortion Ratio | dB (higher = better) | Training loss function |
| Latency | Inference time per 10 ms frame | ms (lower = better) | Real-time feasibility (<10 ms target) |
| Params | Total trainable parameters | Count | Deployment size proxy |

**Key hypothesis:** If DELTA_HASPI (Models 4, 5 vs Model 3) > DELTA_STOI (same comparison), audiogram
conditioning specifically benefits hearing-impaired listeners — not just audio quality in general.

Tool: `pyclarity` computes HASPI/HASQI. Free and open-source.

---

## Tech Stack

```
Core:          PyTorch + torchaudio
Classical:     PyWavelets (Wavelet DWT), pysepm (MMSE-LSA)
DL Models:     Asteroid (ConvTasNet), mamba-ssm (Mamba SSM)
Baselines:     SpeechBrain
Evaluation:    pyclarity (HASPI/HASQI), pystoi (STOI), pesq (PESQ)
Logging:       TensorBoard
Version:       Git + GitHub
Repo:          github.com/Jwanil/hearing-aid-speech-enhancement
```

Install all:
```bash
pip install torch torchaudio asteroid speechbrain pyclarity pystoi pesq PyWavelets mamba-ssm pysepm
```

---

## Project Phases & Directives

| Phase | Name | Directive File | Status |
|-------|------|----------------|--------|
| 0 | Audio/DSP Fundamentals — STFT, spectrograms | `directives/00_dsp_fundamentals.md` | In Progress |
| 1 | Audiology + Audiograms — pyclarity, hearing loss sim | `directives/01_audiology.md` | TODO |
| 2 | Classical Baselines — Wavelet DWT + MMSE-LSA | `directives/02_classical_baselines.md` | TODO |
| 3 | Data Pipeline — TIMIT + Clarity + NOIZEUS | `directives/03_data_pipeline.md` | TODO |
| 4 | 1D CNN (Conv-TasNet) training | `directives/04_1d_cnn_model.md` | TODO |
| 5 | U-Net + Self-Attention + FiLM + CRM + MetricGAN+ | `directives/05_unet_film_model.md` | TODO |
| 5b | Mamba/SSM + FiLM | `directives/05b_mamba_film_model.md` | TODO |
| 6 | Full Evaluation — all 5 models x all metrics x 3 audiogram profiles | `directives/06_evaluation.md` | TODO |
| 7 | Report + Demo | `directives/07_report.md` | TODO |

**Deadline: November 1, 2026**

---

## Key Papers

| Paper | Why It Matters |
|-------|---------------|
| Perez et al., "FiLM" (AAAI 2018) | Core conditioning technique we use |
| Chao et al., "SEMamba" (2024) | SOTA PESQ 3.69 — our Model 5 foundation |
| Gu & Dao, "Mamba" (NeurIPS 2023) | The Mamba SSM architecture |
| Fu et al., "MetricGAN+" (Interspeech 2021) | Our adversarial training framework |
| Diehl et al., Nature Scientific Reports (2023) | Best existing generic HA denoiser — what we beat |
| Kates & Arehart, HASPI v2 (2021) | Our primary evaluation metric |
| Ephraim & Malah (1985) | MMSE-LSA — our classical baseline |
| Donoho & Johnstone (1994) | Wavelet thresholding — our other classical baseline |
| Clarity Challenge (2021–2024) | Dataset + benchmark we evaluate on |
| NeuroAMP (arXiv 2025) | Closest personalization paper (but amplification, not denoising) |

---

## Team

- **Jwanil Modi (23BIT194)** — Lead developer, project ideation, faculty coordination
- **Namya Shah (23BIT027)** — Co-developer, data pipeline, evaluation

**Workflow:** VS Code + Git/GitHub. Each person logs sessions separately in `shared_context.md`.

---

## For Agent Sessions

When starting a new session using this file as context:
1. Read `context.md` — current task state, open tasks, what was last done
2. Read `shared_context.md` — what Jwanil and Namya have each done (separate logs)
3. Read the relevant `directives/` file for the current phase
4. Begin exactly where the logs say to begin
5. **After every action:** update `context.md` and the correct person's section in `shared_context.md`
6. Never re-discuss decisions already marked Final in `shared_context.md` — they are locked
