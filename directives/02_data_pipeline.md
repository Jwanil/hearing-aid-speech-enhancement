# Directive 03 — Dataset Collection, Labeling & Data Pipeline

**Phase:** 3 (but dataset collection starts FIRST, before Phase 2 — faculty instruction)
**Goal:** Collect raw .wav files from chosen datasets, organize and label them, normalize everything
to a common format, then build the PyTorch DataLoader that feeds training data to all models.
**Estimated time:** 1.5 weeks (Sep 7–20, 2026)
**Lead:** Namya (pipeline) + Jwanil (dataset organization)
**Output:** A clean, labeled data directory + a fast, bug-free DataLoader.

---

## PART A — Dataset Collection & Labeling (Do This First)

> Faculty instruction: collect .wav files first, label them, store them before writing any model code.

### Step 1: Folder Structure

Create this folder structure in the project root:

```
data/
├── raw/
│   ├── clean/          <- original clean speech .wav files (from TIMIT, NOIZEUS clean set)
│   ├── noise/          <- noise-only .wav files (from MUSAN, ESC-50)
│   └── noisy/          <- pre-mixed noisy speech (from NOIZEUS, Clarity)
│
├── processed/
│   ├── clean/          <- clean speech after standardization (16kHz, mono, [-1,1])
│   ├── noise/          <- noise clips after standardization
│   └── noisy/          <- noisy speech after standardization
│
└── metadata/
    ├── train_manifest.csv    <- training file list with labels
    ├── val_manifest.csv      <- validation file list
    └── test_manifest.csv     <- test file list (NOIZEUS evaluation)
```

### Step 2: Download Sources

| Dataset | What to Download | Where to Download | Size |
|---------|-----------------|-------------------|------|
| **NOIZEUS** | All noisy .wav files + clean reference | ecs.utdallas.edu/loizou/speech/noizeus/ | ~100 MB |
| **VoiceBank-DEMAND** ⭐ | clean_trainset_28spk_wav + noisy_trainset_28spk_wav + testsets | **datashare.ed.ac.uk/items/6ed35425-bf14-4d2b-93a1-0a4984952757** | ~10 GB |
| **MUSAN** | Noise subset only (not music) | openslr.org/17/ | ~10 GB, use noise/ subfolder only |
| **ESC-50** | All 50 noise categories | github.com/karolpiczak/ESC-50 | ~600 MB |
| **Clarity CEC2** | Register then download | claritychallenge.org | ~20 GB |
| **TIMIT** | Full corpus | LDC / college library access | ~400 MB |

> **Download order:** NOIZEUS first (tiny, instant) → VoiceBank-DEMAND second (gold standard) → rest when ready.

---

### About VoiceBank-DEMAND (Edinburgh DataShare)

**Full name:** Noisy speech database for training speech enhancement algorithms and TTS models
**Authors:** Valentini-Botinhao et al., University of Edinburgh (2017)
**Paper:** "Speech Enhancement for a Noise-Robust Text-to-Speech Synthesis System using Deep Recurrent Neural Networks", Interspeech 2016

This is the **single most cited speech enhancement dataset in the world**. Nearly every major paper
(Conv-TasNet, MetricGAN, CMGAN, SEMamba) reports results on this dataset. This means:
- Our results on VoiceBank-DEMAND can be **directly compared to published numbers** without any
  ambiguity about dataset differences
- PESQ, STOI, and SI-SDR scores from our models can be placed side-by-side with those in papers

**What is in it:**
| Split | Speakers | Utterances | Noise types | SNR levels |
|-------|---------|-----------|------------|-----------|
| Train | 28 speakers | 11,572 utterances | 10 types (from DEMAND database) | 0, 5, 10, 15 dB |
| Test | 2 speakers | 824 utterances | 5 different noise types | 2.5, 7.5, 12.5, 17.5 dB |

**Noise types used:** cafeteria, restaurant, office, public transport, street, etc. (from the DEMAND noise database — real-world, recorded with microphone arrays)

**Key property:** Train and test noise types are DIFFERENT — the model must generalize, not memorize.

**Sampling rate:** 48 kHz (must be resampled to 16 kHz before use — this is standard practice)

**Why this is better than NOIZEUS for training:**
- NOIZEUS has only 30 clean utterances — too small to train a deep learning model
- VoiceBank-DEMAND has 11,572 utterances — large enough for CNN and U-Net training
- VoiceBank-DEMAND is the industry-standard benchmark — NOIZEUS is for classical method evaluation

**Download directly from:** https://datashare.ed.ac.uk/items/6ed35425-bf14-4d2b-93a1-0a4984952757

Files to download (the ones in the file list on that page):
- `clean_trainset_28spk_wav.zip` — clean speech (train split)
- `noisy_trainset_28spk_wav.zip` — noisy speech (train split)  
- `clean_testset_wav.zip` — clean speech (test split)
- `noisy_testset_wav.zip` — noisy speech (test split)

> **Note:** The noisy files in VoiceBank-DEMAND are pre-mixed (not raw clean + noise separately).
> You do NOT need to mix them yourself — they are ready to use as-is.
> For the DataLoader, use clean as the training target and noisy as the model input.



### Step 3: Labeling — the Metadata CSV

Each .wav file needs a row in the CSV with these columns:

```
filename, clean_file, noise_type, snr_db, duration_s, sample_rate, audiogram_profile, split
```

Example rows:
```
noisy/sp01_babble_sn0.wav, clean/sp01.wav, babble, 0, 2.5, 8000, flat_40, test
noisy/sp01_babble_sn5.wav, clean/sp01.wav, babble, 5, 2.5, 8000, flat_40, test
```

For NOIZEUS, the noise type and SNR are encoded in the filename (e.g., `sp01_babble_sn0.wav` =
speaker 1, babble noise, SNR 0 dB). Write a script to auto-parse these.

Script: `execution/03a_build_metadata.py`

---

## PART B — Standardization (Normalize All .wav Files)

> Faculty instruction: transform all .wav files to the same time and frequency range before using.

This is called **preprocessing** and it MUST be done before any model training. Your datasets have:

| Problem | Example | Fix |
|---------|---------|-----|
| Different sampling rates | NOIZEUS=8kHz, TIMIT=16kHz, Clarity=16kHz | Resample everything to **16,000 Hz** |
| Different amplitudes | Some files are very quiet (-40 dBFS), some loud (-3 dBFS) | Normalize to peak amplitude **[-1.0, +1.0]** |
| Different lengths | 1.5s sentence vs 10s sentence | Pad to target length OR use chunked loading |
| Stereo vs mono | Some Clarity files are multi-channel | Convert to **mono** (average channels) |
| Different bit depths | 16-bit PCM vs 32-bit float | Load as float32 uniformly |

### Standardization Script: `execution/03b_standardize_audio.py`

```python
import torchaudio
import torch

TARGET_SR = 16000     # target sampling rate: 16,000 Hz (16 kHz)
TARGET_LEN = 64000    # target length: 4 seconds at 16kHz = 64,000 samples

def standardize(waveform, src_sr):
    """
    Bring any .wav file to the project standard:
    - Mono (1 channel)
    - 16,000 Hz sampling rate
    - Amplitude normalized to [-1, +1]
    - Length = TARGET_LEN samples (pad with silence or trim)
    """
    # 1. Mono conversion
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)   # average all channels -> mono

    # 2. Resample to 16kHz if needed
    if src_sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(src_sr, TARGET_SR)
        waveform = resampler(waveform)

    # 3. Normalize amplitude to [-1, +1]
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak                      # peak normalization

    # 4. Fix length
    length = waveform.shape[1]
    if length > TARGET_LEN:
        waveform = waveform[:, :TARGET_LEN]             # trim if too long
    elif length < TARGET_LEN:
        pad = TARGET_LEN - length
        waveform = torch.nn.functional.pad(waveform, (0, pad))  # zero-pad if too short

    return waveform                                     # shape: (1, 64000)
```

Run this on EVERY file in `data/raw/` and save the output to `data/processed/`.

---

## PART C — Classical Filter Output Preprocessing (Before CNN Input)

> Faculty instruction: preprocess the output of the classical filters before using it as input to the 1D CNN.

When the MMSE-LSA or Wavelet filter processes a noisy file, its output waveform may have:

| Issue | Why It Happens | Fix |
|-------|---------------|-----|
| **Amplitude clipping** | Filter gain > 1.0 at some frames | Re-normalize peak to [-1, +1] |
| **Length mismatch** | Wavelet reconstruction may add/remove 1-2 samples | Trim/pad back to original length |
| **Edge artefacts** | Silence or ringing at start/end | Trim 50ms from each edge and re-pad |
| **DC offset** | Filter introduces a mean shift | Subtract mean: `signal -= signal.mean()` |

### Filter Output Postprocessing Function

```python
def postprocess_filter_output(enhanced, original_length):
    """
    Clean up the output of any classical filter so it is ready
    for CNN input or metric evaluation.
    """
    import numpy as np

    # 1. Remove DC offset
    enhanced = enhanced - enhanced.mean()

    # 2. Fix length to match original input
    if len(enhanced) > original_length:
        enhanced = enhanced[:original_length]
    elif len(enhanced) < original_length:
        enhanced = np.pad(enhanced, (0, original_length - len(enhanced)))

    # 3. Re-normalize amplitude to [-1, +1]
    peak = np.abs(enhanced).max()
    if peak > 0:
        enhanced = enhanced / peak

    return enhanced.astype(np.float32)
```

This function is called AFTER every classical filter produces its output, BEFORE:
- Computing evaluation metrics (HASPI, STOI, PESQ)
- Saving the output .wav file
- Feeding the output as input to the 1D CNN

---

## PART D — PyTorch DataLoader

### What the DataLoader must return per item:

```python
{
    'noisy':    torch.Tensor shape (1, 64000),   # noisy waveform (model input)
    'clean':    torch.Tensor shape (1, 64000),   # clean reference (loss target)
    'audiogram': torch.Tensor shape (6,),        # hearing thresholds [0..1] normalized
    'snr':      float,                           # SNR level used in mixing
    'filename': str,                             # original filename for debugging
}
```

### On-the-fly Mixing (for training splits)

Rather than saving pre-mixed noisy files, the DataLoader mixes on-the-fly each epoch:

```python
def mix_at_snr(clean, noise, target_snr_db):
    """Add noise to clean speech at a specific SNR level."""
    # Energy of clean signal
    clean_energy = (clean ** 2).mean()
    # Energy of noise signal
    noise_energy  = (noise ** 2).mean()
    # Scale noise to hit the target SNR
    scale = (clean_energy / (noise_energy * 10 ** (target_snr_db / 10))) ** 0.5
    return clean + scale * noise
```

This gives effectively infinite training pairs from a finite dataset.

### Execution Scripts

| Script | What It Does |
|--------|-------------|
| `execution/03a_build_metadata.py` | Parse all datasets, build train/val/test CSVs |
| `execution/03b_standardize_audio.py` | Resample + normalize all .wav files to 16kHz, mono, [-1,1] |
| `execution/03c_data_pipeline.py` | PyTorch Dataset class + DataLoader + verification plots |

---

## Tasks (In Order)

- [ ] **Download NOIZEUS** to `data/raw/noisy/` and `data/raw/clean/`
- [ ] **Download MUSAN noise subset** to `data/raw/noise/`
- [ ] **Download VoiceBank-DEMAND** via SpeechBrain (or manually)
- [ ] **Write `execution/03a_build_metadata.py`:** parse filenames → generate `metadata/test_manifest.csv`
- [ ] **Write `execution/03b_standardize_audio.py`:** resample + mono + normalize + fix length → save to `data/processed/`
- [ ] **Verify standardization:** load 5 random files from `data/processed/`, confirm shape `(1, 64000)`, sample rate 16000, values in `[-1, 1]`
- [ ] **Write `execution/03c_data_pipeline.py`:** implement `HearingAidDataset` with on-the-fly mixing
- [ ] **Test DataLoader:** iterate 10 batches, print shapes, plot spectrograms of noisy vs clean

---

## Success Criteria

1. `data/processed/` contains standardized files (16kHz, mono, float32, normalized)
2. `metadata/train_manifest.csv` exists with correct columns
3. DataLoader runs without crashes, returns tensors of shape `(batch, 1, 64000)`
4. Classical filter output postprocessing function tested on 1 NOIZEUS file

---

## Learnings Log

*(Agent: append findings here — note dataset quirks, any files that failed to load, sample rate edge cases)*
