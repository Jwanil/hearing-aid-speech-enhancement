# The Simple Guide — What We Are Actually Building

> **Written for:** Jwanil, Namya Shah, and any AI agent starting fresh on this project.
> **Goal:** Explain everything in plain English. No jargon without a definition. Every short form has its full name the first time.
> **Updated:** August 2026, post-faculty-meeting, post-research-sweep.

---

## Part 1 — The Problem We Are Solving (Plain English)

Imagine you are at a noisy restaurant. Two people with normal hearing can still understand each other across the table. A person wearing a hearing aid often cannot — not because the hearing aid isn't loud enough, but because it is amplifying everything equally: the speech AND the background noise.

**466 million people** worldwide have disabling hearing loss (World Health Organization). The most common complaint from hearing aid users is:

> "I can hear that someone is talking. I just can't understand *what* they are saying in noisy places."

### Why Existing Hearing Aids Fail At This

Current hearing aids use two separate steps:

1. **Noise Reduction** — Try to remove background noise from the microphone signal.
2. **Amplification** — Boost the remaining signal according to the person's hearing profile.

The problem: **Step 1 is generic.** The same noise-reduction algorithm runs for every single user, whether your hearing loss affects high frequencies or low frequencies or everything equally. That makes no sense, because:

- Person A has trouble hearing only high-pitched sounds (consonants like "s", "f", "th")
- Person B has trouble hearing only low-pitched sounds (vowels)
- Person C has uniform loss across all frequencies

All three people get the same noise reduction. Person A needs the high frequencies preserved and boosted. Person B needs the opposite. A single generic algorithm cannot be optimal for everyone.

### Our Solution (One Sentence)

**We build a noise-reduction system that reads each person's hearing test result and adjusts how it cleans the audio specifically for that person's hearing loss.**

---

## Part 2 — The Key Concepts (Defined Simply)

### 2.1 What is Sound, Digitally?

Sound is vibrations in the air. A microphone converts those vibrations into a list of numbers — one number per tiny time slice (usually 1/16,000th of a second for speech). This list of numbers is called a **waveform** or **audio signal**.

```
Waveform looks like:
Time:    0.0001s  0.0002s  0.0003s  0.0004s ...
Value:  [0.021,   0.034,  -0.015,   0.042 , ...]
```

### 2.2 What is a Spectrogram? (Short-Time Fourier Transform / STFT)

A waveform is hard to work with directly — it's just a list of numbers. Instead, we convert it into a **spectrogram**: a 2D picture of sound where:
- The horizontal axis = **Time** (left = early in the recording, right = later)
- The vertical axis = **Frequency** (bottom = low sounds like bass, top = high sounds like a whistle)
- The colour/brightness at each point = **How loud** that frequency is at that moment

Think of it exactly like a **photo of sound**.

The mathematical operation that creates a spectrogram from a waveform is called the **Short-Time Fourier Transform (STFT)**. "Fourier" is the mathematician who invented it. "Short-Time" means we apply it to small overlapping windows of the audio (about 25 milliseconds each).

The reverse operation — turning a spectrogram back into a waveform — is the **Inverse Short-Time Fourier Transform (ISTFT)**.

```
Waveform → [STFT] → Spectrogram (2D image)
Spectrogram → [ISTFT] → Waveform
```

### 2.3 What is an Audiogram?

An audiogram is the result of your hearing test at an audiologist (hearing doctor).

During the test, they play tones at different frequencies and ask "can you hear this?" The quietest sound you can hear at each frequency is your **hearing threshold** — measured in **dB HL (decibels Hearing Level)**.

- **0-25 dB HL** = Normal hearing (you can hear very quiet sounds)
- **25-40 dB HL** = Mild loss (you miss quiet speech)
- **40-70 dB HL** = Moderate loss (you miss normal speech)
- **70+ dB HL** = Severe loss (you only hear loud sounds)

The test is done at 6 standard frequencies: **250 Hz, 500 Hz, 1000 Hz, 2000 Hz, 4000 Hz, 8000 Hz** (from low bass sounds to high treble sounds).

So every person's audiogram is a list of 6 numbers:
```
Example audiogram (severe high-frequency loss):
Frequency:  250Hz  500Hz  1000Hz  2000Hz  4000Hz  8000Hz
Threshold:   10     15      20      45      70      85    (dB HL)
            Normal  Normal  Normal  Mild    Severe  Severe
```

We represent this as a vector (a list): **a = [10, 15, 20, 45, 70, 85]**

### 2.4 What is a Neural Network?

A neural network is a computer program that **learns patterns from examples** instead of being programmed with rules.

Analogy: Teaching a child to recognise dogs. You don't write rules like "4 legs, fur, tail". You just show them 10,000 photos labelled "dog" or "not dog". Eventually they learn.

A neural network works similarly:
- You show it many examples of (noisy speech, clean speech) pairs
- It adjusts millions of internal numbers (called **weights** or **parameters**) to get better at the task
- After training on enough examples, it learns to clean speech it has never seen before

### 2.5 What is a Convolutional Neural Network (CNN)?

A special type of neural network that works with grid-like data (images, spectrograms). Instead of connecting every point to every other point, it uses a small **filter (kernel)** that slides across the input looking for patterns.

Like a magnifying glass scanning across a photo looking for edges, textures, and shapes.

For speech: the filter slides across the spectrogram looking for speech patterns (vowel formants, consonant bursts) and noise patterns.

### 2.6 What is a U-Net?

A specific CNN architecture (designed in 2015 for medical image segmentation) that has become the most popular for speech enhancement. It looks like the letter U:

```
                    [Bottleneck — most compressed]
                   /                              \
          [Encoder 4]                          [Decoder 4]
         /                                              \
  [Encoder 3]                                        [Decoder 3]
 /                                                              \
[Encoder 2]  ← skip connection (shortcut) →             [Decoder 2]
|                                                              |
[Encoder 1]  ← skip connection (shortcut) →             [Decoder 1]
|                                                              |
[Input: Noisy Spectrogram]                        [Output: Mask]
```

- **Encoder** (left side going up): Compresses the spectrogram, finding abstract patterns
- **Bottleneck** (top): The most compressed, highest-level understanding of the audio
- **Decoder** (right side going down): Rebuilds the full spectrogram size, making predictions
- **Skip connections**: Shortcuts that pass fine details from encoder to decoder (so fine details aren't lost in the bottleneck)

### 2.7 What is a Mask?

The model does not directly predict the clean speech. Instead it predicts a **mask** — a number between 0 and 1 for every time-frequency point in the spectrogram:

```
Mask value = 0.0 → "This point is 100% noise, throw it away"
Mask value = 1.0 → "This point is 100% speech, keep it fully"
Mask value = 0.5 → "This point is half speech, half noise"

Enhanced spectrogram = Mask × Noisy spectrogram
```

This is called the **Ideal Ratio Mask (IRM)**. "Ideal" means it's the theoretical best mask if we knew exactly how much speech and noise was at each point.

### 2.8 What is a Complex Ratio Mask (CRM)?

Standard masking (IRM) only works on the **magnitude** (loudness) part of the spectrogram and ignores the **phase** (timing) information. This causes artifacts (weird wobbling sounds called "musical noise").

A **Complex Ratio Mask (CRM)** masks both the magnitude AND the phase simultaneously, producing cleaner, more natural sound. The model predicts 2 numbers per point (real and imaginary parts) instead of 1.

Think of it like this: magnitude tells you HOW LOUD a sound is, phase tells you WHEN the wave peaks. Both matter for reconstructing natural speech.

### 2.9 What is FiLM? (Feature-wise Linear Modulation)

**FiLM** stands for **Feature-wise Linear Modulation** (Perez et al., 2017).

It's a way to inject "side information" (like an audiogram) into the middle of a neural network. Without FiLM, the network processes everyone's audio the same way. With FiLM, the audiogram can change how the network processes audio.

**How it works (simple version):**

```
Normal network layer:
  Input features → [Layer] → Output features

FiLM-conditioned layer:
  Audiogram → [Small network] → Scale (γ) and Shift (β)
  Input features → [Layer] → Raw output → × γ + β → Final output
```

The "Scale and Shift" (called **gamma** γ and **beta** β, Greek letters) effectively turn up or down different feature detectors based on the audiogram. If the audiogram shows severe high-frequency loss, FiLM learns to scale up the features that correspond to high-frequency noise removal.

**Analogy:** Like a music equaliser that automatically adjusts its settings based on your specific hearing test results, every single time.

### 2.10 What is a State Space Model (SSM) — and Mamba?

**State Space Model (SSM)** is a mathematical way of describing systems that have memory — where the output depends on everything that happened before.

**Mamba** (Gu & Dao, 2023) is a specific neural network architecture based on SSMs. It's designed to solve the same problem as Transformers (capturing long-range dependencies in sequences like audio), but much more efficiently.

**The Transformer problem for audio:**
- Transformers work by making every time frame "look at" every other time frame simultaneously
- For 400 frames of audio, that's 400 × 400 = 160,000 comparisons
- The cost grows quadratically (O(T²)) — double the audio length, 4× the computation

**How Mamba works instead:**
- Processes audio left-to-right, one frame at a time
- Maintains a **hidden state** — a compressed memory of everything it has seen so far
- The key innovation: the hidden state is **selective** — it decides what to remember and what to forget based on the current input

```
Standard RNN (old approach): fixed memory rules
  h_new = A × h_old + B × input   (A, B are fixed constants)

Mamba (selective SSM): input-dependent memory rules
  A, B, C = computed from current input
  h_new = A × h_old + B × input   (A, B change based on what we're hearing)
  output = C × h_new
```

**Why this matters for hearing aids:**
- Cost grows linearly (O(T)) — much cheaper than Transformer's O(T²)
- Naturally processes one frame at a time (causal — no "cheating" by looking at future audio)
- Current best performance: **PESQ score of 3.69** on standard benchmarks (beats Transformers)

**Analogy:** A Transformer is like reading an entire book, then re-reading every word while comparing it to every other word. Mamba is like a very smart reader who reads left-to-right, taking smart notes in a notebook, and can refer back to those notes without going back to the beginning.

### 2.11 What is MetricGAN+?

**MetricGAN+** is a training technique that improves how the model learns.

**The problem with standard training:** We train the model by minimising "how wrong" its output is using a mathematical formula (like Mean Squared Error — MSE). But MSE doesn't know what sounds good to a human ear. A small MSE error might sound terrible; a large MSE error might sound fine.

**MetricGAN+'s solution:** Add a second network called a **discriminator** that learns to predict audio quality scores. The main network (generator) is then trained to fool the discriminator into giving high scores.

```
Generator: Noisy speech → Enhanced speech
Discriminator: Enhanced speech → Predicted quality score

Training:
  - Generator tries to maximise the quality score
  - Discriminator tries to accurately predict the real quality score
  They compete — the generator improves until it produces very high-quality audio
```

**The key insight:** At inference (actual use), you throw away the discriminator. It only helps during training. So the final model has zero extra computational cost.

**Our twist:** Instead of optimising for standard audio quality metrics, we will train the discriminator to predict **HASPI (Hearing Aid Speech Perception Index)** — a metric that specifically measures intelligibility for hearing-impaired listeners. This combination (FiLM + HASPI-discriminator) has not been published before.

### 2.12 What is HASPI? (Hearing Aid Speech Perception Index)

**HASPI** stands for **Hearing Aid Speech Perception Index** (Kates & Arehart, 2021). It predicts how well a hearing-impaired person would understand speech after processing.

Unlike standard audio metrics (like STOI — Short-Time Objective Intelligibility, or PESQ — Perceptual Evaluation of Speech Quality), HASPI **takes the person's audiogram as an input**. It simulates how the impaired cochlea processes the audio and computes a score (0 to 1, higher is better) predicting intelligibility.

### 2.13 What is HASQI? (Hearing Aid Speech Quality Index)

**HASQI** stands for **Hearing Aid Speech Quality Index** (Kates & Arehart, 2014). Same idea as HASPI but measures **quality** (naturalness, absence of distortion) rather than intelligibility.

### 2.14 What is Knowledge Distillation?

**Knowledge Distillation** is a technique for making a small, fast model learn from a large, accurate model.

- **Teacher model:** Large, accurate, slow (e.g., our Mamba model)
- **Student model:** Small, fast, deployable (e.g., a tiny U-Net that meets the <10ms latency constraint)

The student is trained not just on the data, but also to **mimic the teacher's internal behaviour** — not just the final output, but also the intermediate features. This lets the student "punch above its weight class."

**Analogy:** A senior expert (teacher) guides a junior intern (student). The intern doesn't just copy the expert's answers — they learn the expert's reasoning process.

### 2.15 What is STOI? (Short-Time Objective Intelligibility)

**STOI** stands for **Short-Time Objective Intelligibility** (Taal et al., 2011). A number between 0 and 1 predicting how well a normal-hearing person would understand the speech. We use it alongside HASPI because HASPI requires an audiogram — STOI lets us compare with the general speech enhancement literature.

### 2.16 What is SI-SDR? (Scale-Invariant Signal-to-Distortion Ratio)

**SI-SDR** stands for **Scale-Invariant Signal-to-Distortion Ratio**. A number in decibels (dB) measuring overall signal quality. "Scale-invariant" means it doesn't penalise the model for making the audio louder or quieter than the reference — only the distortion matters.

Higher = better. Typically ranges from -5 to 25 dB for good speech enhancement models.

### 2.17 What are ERB Bands? (Equivalent Rectangular Bandwidth)

**ERB** stands for **Equivalent Rectangular Bandwidth**. 

The human ear doesn't process all frequencies equally. It has better frequency resolution (more detailed hearing) at low frequencies, and worse resolution at high frequencies. The ERB scale mathematically captures this non-uniform resolution.

When we split the audio spectrogram into ERB bands, we're splitting it the way the human ear naturally does — more bands for low frequencies (where speech formants live), fewer bands for high frequencies. This is more efficient AND more aligned with how hearing-impaired listeners actually perceive sound.

---

## Part 3 — What We Are Building (The Architecture)

We are building **5 models** and comparing them side by side:

### Model 1: Wavelet Denoising (Classical Baseline)

**Full name:** Discrete Wavelet Transform (DWT) denoising using soft thresholding.

**What it does:** Converts the audio into a wavelet representation (similar to a spectrogram but with adaptive time-frequency resolution), sets small coefficients (likely noise) to zero, converts back.

**Why we include it:** Best classical approach for non-stationary noise. More sophisticated than the basic Wiener filter. Used in actual hearing aids.

**Resources to learn:**
- PyWavelets library tutorial: https://pywavelets.readthedocs.io/
- YouTube: "Wavelet Transform explained" by Matlab (15 min)
- Paper: Donoho & Johnstone, "Ideal Spatial Adaptation by Wavelet Shrinkage" (1994)

### Model 2: MMSE-LSA Filter (Classical Baseline)

**Full name:** Minimum Mean Square Error - Log Spectral Amplitude (MMSE-LSA) estimator.

**What it does:** Estimates the ratio of speech to noise at each time-frequency point using statistical methods, then applies a smoothed gain to suppress noise without creating "musical noise" artifacts. The industry standard in hearing aids today.

**Why we include it:** This is what commercial hearing aids actually use. Beating it with DL is meaningful. The LSA (Log Spectral Amplitude) variant is preferred over the original MMSE because it sounds more natural.

**Resources to learn:**
- Paper: Ephraim & Malah, "Speech Enhancement Using a Minimum Mean Square Error Log-Spectral Amplitude Estimator" (1985)
- Book: Loizou, "Speech Enhancement: Theory and Practice" — Chapter 6
- Python implementation: Look up `pysepm` library or write from scratch following the paper

### Model 3: 1D Convolutional Neural Network (Conv-TasNet style)

**Full name:** 1-Dimensional Convolutional Neural Network, specifically a Temporal Convolutional Network (TCN) in the style of Conv-TasNet (Convolution-based Time-domain Audio Separation Network).

**What it does:** Works directly on the raw audio waveform (not spectrogram). Learns its own "frequency decomposition" instead of using STFT. Uses dilated depthwise-separable convolutions to capture both short and long patterns in time.

**Why we include it:** The lightest DL approach. Naturally lowest latency. Represents the "waveform approach" vs our "spectrogram approach".

**Resources to learn:**
- Paper: Luo & Mesgarani, "Conv-TasNet" (2019) — readable
- Code: Asteroid library (pip install asteroid) — has Conv-TasNet built-in
- YouTube: "Conv-TasNet explained" — several good walkthroughs on YouTube

### Model 4: U-Net + Attention + FiLM (Our Core Contribution)

**Full name:** U-Net with Self-Attention mechanism, Feature-wise Linear Modulation (FiLM) conditioning on audiogram, Complex Ratio Mask (CRM) prediction, trained with MetricGAN+ discriminator.

**What it does:**
1. Takes noisy speech spectrogram + person's audiogram as inputs
2. Encoder compresses the spectrogram into abstract features
3. Self-attention in the bottleneck lets every part of the audio look at every other part
4. FiLM adjusts the processing based on the audiogram
5. Decoder predicts a Complex Ratio Mask (both magnitude AND phase)
6. Mask applied to noisy spectrogram → enhanced spectrogram → ISTFT → enhanced speech

**Why it's the core contribution:** This personalises noise reduction per person's hearing profile. Generic DNNs don't do this.

**Resources to learn:**
- U-Net paper: Ronneberger et al. (2015) — original image segmentation paper
- FiLM paper: Perez et al. (2017) — very readable
- DCCRN paper: Hu et al. (2020) — shows how to do complex masking in a U-Net

### Model 5: Mamba / State Space Model (SSM) + FiLM

**Full name:** State Space Model (SSM) architecture using Mamba selective scan mechanism, with Feature-wise Linear Modulation (FiLM) audiogram conditioning.

**What it does:** Same task as Model 4 but replaces the CNN/Attention layers with Mamba SSM blocks. Processes the spectrogram sequentially with a selective memory mechanism. Currently achieves the best published performance (PESQ 3.69) while using linear computational complexity.

**Why we include it:** The current state-of-the-art for speech enhancement. More accurate than Transformers, faster than Transformers. The closest to being actually deployable on a real hearing aid chip.

**Resources to learn:**
- Paper: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
- Blog post: "The Annotated Mamba" — detailed walkthrough with code
- Paper: Chao et al., "SEMamba" (2024) — the speech enhancement application
- Code: SEMamba GitHub repository — working PyTorch implementation

---

## Part 4 — The Complete System Pipeline

```
                         ┌───────────────────────────────┐
                         │      INPUT                    │
                         │  Noisy Speech (microphone)    │
                         └──────────────┬────────────────┘
                                        │
                                        ▼
                         ┌───────────────────────────────┐
                         │  Step 1: STFT                 │
                         │  (Short-Time Fourier Transform)│
                         │  Converts waveform → spectrogram│
                         └──────────────┬────────────────┘
                                        │
                              Noisy Spectrogram
                              (2D image of sound)
                                        │
                 ┌──────────────────────┼──────────────────────┐
                 │                      │                      │
                 │              ┌───────┴──────────────┐       │
                 │              │   AUDIOGRAM INPUT    │       │
                 │              │   a = [10,15,20,     │       │
                 │              │        45, 70, 85]   │       │
                 │              └───────────────────────┘       │
                 │                                              │
                 ▼                                              │
┌───────────────────────────────────────┐                      │
│  MODEL (U-Net or Mamba)               │                      │
│                                       │◄─────────────────────┘
│  Encoder: compress spectrogram        │  FiLM injects audiogram
│  Attention: global context            │  into the middle of the model
│  FiLM: personalise per audiogram      │
│  Decoder: predict mask                │
└──────────────────────┬────────────────┘
                       │
                  Complex Ratio Mask
                  (values 0→1 per point)
                       │
                       ▼
         ┌─────────────────────────┐
         │   Mask × Noisy Spec.   │
         │   (point-wise multiply)│
         └──────────┬──────────────┘
                    │
          Enhanced Spectrogram
                    │
                    ▼
         ┌─────────────────────────┐
         │   ISTFT                 │
         │  (Inverse STFT)         │
         │  Spectrogram → Waveform │
         └──────────┬──────────────┘
                    │
                    ▼
         ┌─────────────────────────┐
         │   OUTPUT                │
         │   Enhanced Speech       │
         │   (cleaner, more        │
         │    intelligible)        │
         └─────────────────────────┘
```

---

## Part 5 — How We Evaluate (Metrics)

We compare all 5 models on the same test audio using these measurements:

| Metric | Full Name | What It Measures | Range | Better = |
|--------|-----------|------------------|-------|----------|
| **HASPI** | Hearing Aid Speech Perception Index | Intelligibility for hearing-impaired listeners (takes audiogram as input) | 0 to 1 | ↑ Higher |
| **HASQI** | Hearing Aid Speech Quality Index | Quality / naturalness for hearing-impaired listeners (takes audiogram as input) | 0 to 1 | ↑ Higher |
| **STOI** | Short-Time Objective Intelligibility | Intelligibility for normal-hearing listeners | 0 to 1 | ↑ Higher |
| **SI-SDR** | Scale-Invariant Signal-to-Distortion Ratio | Overall signal quality | dB | ↑ Higher |
| **PESQ** | Perceptual Evaluation of Speech Quality | Perceptual quality (phone call standard) | 1 to 4.5 | ↑ Higher |
| **Latency** | — | Processing delay per frame | milliseconds | ↓ Lower |
| **Params** | — | Number of model weights | Millions | ↓ Lower |

**The hearing aid real-time constraint:** Latency must be under **10 milliseconds (ms)**. Any delay longer than this causes the person to hear both the delayed processed sound AND the direct sound through the ear, causing a confusing echo.

**Our key claim:** HASPI improvement should be **larger** for our personalised models (4 and 5) than for the generic model (3). This would prove that audiogram conditioning helps specifically for hearing-impaired listeners — not just sound quality in general.

---

## Part 6 — Datasets (Where We Get Our Data)

| Dataset | Full Name | What It Contains | Why We Use It |
|---------|-----------|-----------------|---------------|
| **TIMIT** | Texas Instruments / MIT | 6,300 clean speech recordings from 630 speakers | Clean speech source for training. Has phoneme labels for detailed analysis. |
| **NOIZEUS** | NOIsy speech corpus (University of Texas Dallas) | 30 sentences with 8 real-world noises at 4 signal-to-noise levels | Standard evaluation benchmark in speech enhancement literature |
| **Clarity CEC2/3** | Clarity Enhancement Challenge 2/3 | Noisy speech specifically for hearing aid research, includes real listener audiograms | Primary training dataset. Purpose-built for exactly our problem. |
| **VoiceBank-DEMAND** | VoiceBank (speakers) + DEMAND (noise) | 28 speakers + 8 noise types | Secondary training / sanity check. Most common SE benchmark in literature. |
| **MUSAN** | Music, Speech, and Noise | ~900 noise recordings across types | Additional noise sources for data augmentation during training |

**Data augmentation strategy:** Instead of pre-creating fixed noisy files, we mix clean speech + noise **on-the-fly** during training at random signal-to-noise ratios between -5 dB and +10 dB. This gives us effectively infinite training variety.

---

## Part 7 — Tools and Libraries

| Tool | What It Is | Why We Use It |
|------|-----------|---------------|
| **Python** | Programming language | Everything is written in Python |
| **PyTorch** | Python library for neural networks | Build and train all our models |
| **torchaudio** | Extension of PyTorch for audio | STFT/ISTFT, audio loading |
| **SpeechBrain** | High-level speech processing toolkit | Pretrained baselines, MetricGAN+ implementation |
| **Asteroid** | Source separation toolkit | Conv-TasNet reference implementation |
| **pyclarity** | Python library for Clarity Challenge | HASPI and HASQI computation, hearing loss simulation |
| **pywavelets (pywt)** | Wavelet library | Discrete Wavelet Transform for Model 1 |
| **librosa** | Audio analysis library | Spectrogram visualisation, audio analysis |
| **matplotlib** | Plotting library | All our graphs and visualisations |
| **TensorBoard** | Training visualisation | Watch training progress in real-time |
| **git + GitHub** | Version control | Sync code between Jwanil and Namya |

---

## Part 8 — Timeline to November 1, 2026

| Phase | What We Build | Duration | Dates | Lead |
|-------|--------------|----------|-------|------|
| **Phase 0** | DSP (Digital Signal Processing) Fundamentals — STFT, spectrograms, audio basics | 1 week | Aug 15 – Aug 22 | Both |
| **Phase 1** | Audiology + Audiogram generation — simulate hearing loss | 1 week | Aug 23 – Aug 29 | Both |
| **Phase 2** | Classical Baselines — Wavelet denoising + MMSE-LSA filter | 1 week | Aug 30 – Sep 6 | Jwanil |
| **Phase 3** | Data Pipeline — PyTorch DataLoader with on-the-fly mixing | 2 weeks | Sep 7 – Sep 20 | Namya |
| **Phase 4** | 1D CNN Model — Conv-TasNet style waveform denoiser | 2 weeks | Sep 21 – Oct 4 | Both |
| **Phase 5** | U-Net + Attention + FiLM — Core contribution with complex masking | 2 weeks | Oct 5 – Oct 18 | Both |
| **Phase 5b** | Mamba + FiLM — State Space Model based personalised model | 1 week | Oct 19 – Oct 25 | Jwanil |
| **Phase 6** | Full Evaluation — HASPI, HASQI, STOI, SI-SDR, latency across all 5 models | 3 days | Oct 26 – Oct 28 | Both |
| **Phase 7** | Report + Audio Demo — Final document and before/after audio samples | 3 days | Oct 29 – Oct 31 | Both |
| 🎯 **DEADLINE** | Submit final project | — | **Nov 1, 2026** | — |

---

## Part 9 — Resources to Study (Phase by Phase)

### Phase 0 — Audio/DSP (Digital Signal Processing)
- **Video:** 3Blue1Brown — "But what is a Fourier series?" (YouTube, 20 min) — visually explains frequency decomposition
- **Video:** YouTube — "STFT (Short-Time Fourier Transform) explained" — multiple good options
- **Code:** librosa tutorial (librosa.org) — hands-on spectrogram generation
- **Book:** Chapter 2-4 of Loizou "Speech Enhancement" (available via college library)

### Phase 1 — Audiology
- **Website:** WHO (World Health Organization) hearing loss page — overview statistics
- **Video:** YouTube — "How to read an audiogram" (audiology channel, 10 min)
- **Code:** pyclarity documentation — hearing loss simulation tutorial
- **Paper:** Kates & Arehart, "HASPI Version 2" (2021) — skim the introduction

### Phase 2 — Classical Baselines
- **Book:** Loizou "Speech Enhancement" Chapter 6 (Wiener/MMSE) + Chapter 9 (Wavelet)
- **Code:** PyWavelets documentation (pywavelets.readthedocs.io)
- **Code:** pysepm library — reference implementations of classical SE algorithms
- **Paper:** Donoho & Johnstone, "Ideal Spatial Adaptation by Wavelet Shrinkage" (1994)

### Phase 3 — Data Pipeline
- **Tutorial:** PyTorch official tutorial — "Writing Custom Datasets, DataLoaders and Transforms"
- **Tutorial:** SpeechBrain dynamic mixing tutorial (docs.speechbrain.com)
- **Code:** torchaudio documentation — loading and processing audio files

### Phase 4 — 1D CNN
- **Paper:** Luo & Mesgarani, "Conv-TasNet" (2019) — ICASSP — read the full paper
- **Code:** Asteroid library tutorial — pip install asteroid, run their Conv-TasNet example
- **Tutorial:** YouTube — "Conv-TasNet explained" — search for this, several good options

### Phase 5 — U-Net + FiLM
- **Paper:** Ronneberger, "U-Net" (2015) — the original 8-page paper, very readable
- **Paper:** Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer" (2017)
- **Paper:** Hu et al., "DCCRN" (2020) — shows complex masking in a U-Net
- **Code:** Look up "pytorch U-Net speech enhancement github" — several clean implementations

### Phase 5b — Mamba
- **Paper:** Gu & Dao, "Mamba: Linear-Time Sequence Modeling" (2023) — read Section 1-3
- **Blog:** "The Annotated Mamba" — Google this, line-by-line Python explanation
- **Paper:** Chao et al., "SEMamba" (2024) — speech enhancement application
- **Code:** SEMamba GitHub — working implementation to adapt

### Phase 6 — Evaluation
- **Code:** pyclarity Python package — `pip install pyclarity`, compute HASPI/HASQI
- **Code:** pystoi Python package — `pip install pystoi`, compute STOI
- **Code:** torch.profiler — measure model latency and FLOP count

### Phase 7 — Report
- **Template:** Use `docs/project_overview.md` as structural reference
- **Audio:** Use pydub or scipy to save before/after audio samples as .wav files

---

## Part 10 — The Novelty of Our Contribution

Here is what makes our project different from anything published:

1. **FiLM conditioning for noise reduction** (not amplification): NeuroAMP (2025) conditions on audiograms for amplification. We do it for the upstream noise reduction step — this combination has not been published.

2. **HASPI-optimised discriminator**: Training MetricGAN+ to predict HASPI (instead of PESQ) means we're directly optimising for what hearing-impaired listeners will experience. This is a novel training approach.

3. **Mamba + FiLM**: The current SOTA Mamba architecture (SEMamba) does not include audiogram conditioning. Adding FiLM to SEMamba is novel.

4. **Three-way architecture comparison with HA metrics**: The systematic comparison of Wavelet/MMSE/1D-CNN/U-Net-FiLM/Mamba-FiLM using HASPI/HASQI has not been published in this form.

---

## Part 11 — What Success Looks Like

By November 1, 2026, we will have:

- [ ] 5 trained models (2 classical + 3 deep learning)
- [ ] A results table showing all models compared on HASPI, HASQI, STOI, SI-SDR, PESQ, latency
- [ ] Audio demonstrations: for 3 audiogram profiles (mild, moderate, severe), playable before/after audio files
- [ ] Honest discussion of the latency constraint (which models meet <10ms, which don't, what knowledge distillation could do)
- [ ] A written report (this document + the full academic report)
- [ ] Fully reproducible code (anyone can run `python execution/XX_script.py` to reproduce results)

**The single most important result:** If our personalised model (Model 4 or 5) shows a larger HASPI improvement than the generic model (Model 3) — particularly for non-uniform audiograms like severe high-frequency loss — we have proved our core hypothesis and the project is a success.
