# Directive 02 — Classical Baselines (MMSE-LSA Filter first, then Wavelet Denoising)

**Phase:** 2
**Goal:** Implement two classical (non-deep-learning) speech enhancement baselines.
These are the reference points every deep learning model must beat.
**Estimated time:** 1 week (Aug 30 – Sep 6, 2026)
**Lead:** Jwanil
**Output:** Two working classical denoisers with documented metric scores on NOIZEUS.

> **Faculty instruction:** Implement MMSE filter FIRST, then Wavelet filter.
> Reason: MMSE teaches you noise estimation and gain functions — the core pattern behind
> ALL enhancement algorithms. Wavelet is easier after MMSE is understood.

---

## Why Two Classical Baselines (Not One)?

The faculty recommended replacing the basic Wiener filter with more sophisticated classical approaches.

| Baseline | Technique | Why It's Included |
|---|---|---|
| **MMSE-LSA Filter** (do first) | Minimum Mean Square Error – Log Spectral Amplitude | The industry standard in commercial hearing aids today (Oticon, Phonak, Signia). Beating it is a meaningful result. |
| **Wavelet Denoising** (do second) | Discrete Wavelet Transform (DWT) + soft thresholding | Best classical approach for non-stationary noise. Adaptive time-frequency resolution. Used in real hearing aids. |

---

## PRE-REQUISITE: Standardized Audio

Before running any filter, your audio MUST be in the standard project format:
- **Sampling rate:** 16,000 Hz (16 kHz)
- **Channels:** Mono (1 channel)
- **Amplitude:** Normalized to peak [-1.0, +1.0]
- **Length:** 4 seconds = 64,000 samples (pad/trim as needed)

See `directives/03_data_pipeline.md` → Part B for the standardize() function.
Run `execution/03b_standardize_audio.py` on NOIZEUS files before this phase.

---

## MODEL 1 — MMSE-LSA Filter (Implement First)

### What It Is

Introduced by Ephraim & Malah (1985). The algorithm running in most commercial hearing aids today.

Key improvements over the basic Wiener filter:
- Uses a **decision-directed approach** to estimate the a priori (prior) SNR — smooths gain over time
- Operates on **log spectral amplitude** — better matches human auditory perception
- Dramatically reduces "musical noise" — the annoying tonal artifacts that plague simpler approaches

### Background: The Decision-Directed SNR Estimate

```
For every STFT (Short-Time Fourier Transform) frame n and frequency bin k:

1. Compute instantaneous SNR (gamma):
   gamma(k,n) = |Y(k,n)|^2 / lambda_n(k,n)
   where Y = noisy STFT, lambda_n = noise power estimate

2. Update a priori SNR (xi) using decision-directed approach:
   xi(k,n) = alpha * [A_hat^2(k,n-1) / lambda_n(k,n)] + (1-alpha) * max(gamma(k,n) - 1, 0)
   where alpha = 0.98, A_hat = previous frame's enhanced amplitude

3. Compute MMSE-LSA gain:
   v = xi(k,n) * gamma(k,n) / (1 + xi(k,n))
   G(k,n) = xi(k,n) / (1 + xi(k,n)) * exp(0.5 * E1(v))
   where E1(v) is the exponential integral (use scipy.special.exp1)

4. Apply gain: A_hat(k,n) = G(k,n) * |Y(k,n)|

5. Reconstruct: use phase from noisy signal + ISTFT
```

### Implementation Steps

```python
import numpy as np
import scipy
from scipy.special import exp1

def mmse_lsa_filter(noisy_waveform, sr=16000, frame_ms=25, hop_ms=10, alpha=0.98):
    """
    Apply MMSE-LSA speech enhancement.
    noisy_waveform: np.array, shape (N,), float32, values in [-1, +1]
    sr: sample rate (must be 16000 after standardization)
    Returns: enhanced waveform, same shape and range.
    """
    # STFT parameters
    n_fft  = int(sr * frame_ms / 1000)    # e.g., 25ms = 400 samples
    hop    = int(sr * hop_ms  / 1000)     # e.g., 10ms = 160 samples
    window = np.hanning(n_fft)

    # Compute STFT -> complex spectrogram
    # Use scipy.signal.stft or librosa.stft
    # Shape: (freq_bins, time_frames)
    f, t, Y = scipy.signal.stft(noisy_waveform, fs=sr, window=window,
                                 nperseg=n_fft, noverlap=n_fft-hop)

    mag   = np.abs(Y)               # magnitude spectrogram
    phase = np.angle(Y)             # phase (we will keep this from noisy signal)

    n_freq, n_frames = mag.shape

    # Noise estimation: use first 6 frames (first ~60ms) as initial noise estimate
    noise_est = (mag[:, :6] ** 2).mean(axis=1, keepdims=True)   # shape: (freq_bins, 1)

    xi    = np.ones((n_freq, 1))   # a priori SNR initialization
    A_hat = np.zeros((n_freq, 1))  # previous enhanced amplitude
    G_out = np.zeros_like(mag)     # gain output

    for n in range(n_frames):
        Y_n     = mag[:, n:n+1]            # current frame magnitude
        gamma_n = Y_n**2 / (noise_est + 1e-10)   # instantaneous SNR

        # Decision-directed a priori SNR update
        if n == 0:
            xi_n = alpha * np.ones_like(gamma_n) + (1-alpha) * np.maximum(gamma_n - 1, 0)
        else:
            xi_n = alpha * (A_hat**2 / (noise_est + 1e-10)) + (1-alpha) * np.maximum(gamma_n - 1, 0)

        # MMSE-LSA gain
        v       = xi_n * gamma_n / (1 + xi_n)
        v       = np.clip(v, 1e-10, 700)    # numerical safety
        G_n     = xi_n / (1 + xi_n) * np.exp(0.5 * exp1(v))
        G_n     = np.clip(G_n, 0.0, 1.0)   # gain in [0, 1]

        A_hat   = G_n * Y_n                 # enhanced amplitude
        G_out[:, n:n+1] = G_n

        # Update noise estimate (minimum statistics — simple version)
        # Full min-statistics tracking: track minimum power over last 1.5s window
        noise_est = 0.98 * noise_est + 0.02 * np.minimum(Y_n**2, noise_est * 10)

    # Apply gain and reconstruct
    enhanced_mag = G_out * mag
    enhanced_Y   = enhanced_mag * np.exp(1j * phase)   # restore noisy phase
    _, enhanced  = scipy.signal.istft(enhanced_Y, fs=sr, window=window,
                                       nperseg=n_fft, noverlap=n_fft-hop)

    return enhanced.astype(np.float32)
```

---

## MODEL 2 — Wavelet DWT Denoising (Implement Second)

### What It Is

DWT (Discrete Wavelet Transform) converts a 1D signal into coefficients at multiple time-scale levels.
Speech has energy at specific scales; noise spreads uniformly. Thresholding small coefficients
removes noise while preserving speech structure.

**Reference:** Donoho & Johnstone (1994) — universal threshold, soft thresholding.

### Algorithm

```
1. Compute DWT(signal) -> wavelet coefficients at levels 1..L
   (like a spectrogram but adaptive: fine time resolution at high freq,
   fine freq resolution at low freq — matches how speech is structured)

2. Estimate noise sigma from finest detail level:
   sigma = median(|coefficients at level 1|) / 0.6745
   (the 0.6745 is the MAD-to-sigma conversion for Gaussian noise)

3. Compute universal threshold:
   lambda = sigma * sqrt(2 * log(N))    where N = signal length

4. Apply SOFT threshold to all detail coefficients (NOT the approximation):
   coeff_new = sign(coeff) * max(|coeff| - lambda, 0)
   (soft thresholding reduces coefficients smoothly, not abruptly)

5. Inverse DWT -> denoised signal
```

### Implementation

```python
import pywt
import numpy as np

def wavelet_denoise(waveform, wavelet='db8', level=5, threshold_mode='soft'):
    """
    Apply Wavelet DWT denoising.
    waveform: np.array, float32, values in [-1, +1], already at 16kHz
    wavelet: 'db8' (Daubechies-8) is standard for speech
    level: 5 for 16kHz (covers 250Hz to 8000Hz speech range)
    Returns: denoised waveform, same shape.
    """
    # Step 1: Decompose
    coeffs = pywt.wavedec(waveform, wavelet=wavelet, level=level)
    # coeffs[0]        = approximation (coarse, keep this untouched)
    # coeffs[1..level] = detail coefficients at each level (fine to coarse)

    # Step 2: Estimate noise sigma from finest detail level
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745

    # Step 3: Universal threshold
    N      = len(waveform)
    lam    = sigma * np.sqrt(2 * np.log(N))

    # Step 4: Soft threshold all detail coefficients (skip coeffs[0])
    denoised_coeffs = [coeffs[0]]                # keep approximation unchanged
    for detail_level in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(detail_level, lam, mode='soft'))

    # Step 5: Reconstruct
    denoised = pywt.waverec(denoised_coeffs, wavelet)

    # Safety: waverec may give 1-2 extra samples due to boundary handling
    denoised = denoised[:len(waveform)]

    return denoised.astype(np.float32)
```

---

## Classical Filter Output Postprocessing (Apply to BOTH Filters)

After either filter runs, ALWAYS postprocess the output before:
- Computing metrics (HASPI, STOI)
- Saving the .wav file
- Using as input for the 1D CNN

```python
def postprocess_filter_output(enhanced, original_length):
    """
    Standardize classical filter output for downstream use.
    Call this after mmse_lsa_filter() or wavelet_denoise().
    """
    import numpy as np

    # 1. Remove DC offset (filter may introduce a mean shift)
    enhanced = enhanced - enhanced.mean()

    # 2. Fix length (wavelet reconstruction may add/remove 1-2 samples)
    if len(enhanced) > original_length:
        enhanced = enhanced[:original_length]
    elif len(enhanced) < original_length:
        enhanced = np.pad(enhanced, (0, original_length - len(enhanced)))

    # 3. Re-normalize to [-1, +1]
    peak = np.abs(enhanced).max()
    if peak > 1e-6:                     # don't divide by near-zero (silence)
        enhanced = enhanced / peak

    return enhanced.astype(np.float32)
```

---

## Execution Scripts

| Script | What It Does |
|--------|-------------|
| `execution/04_mmse_lsa_filter.py` | Loads standardized noisy .wav, applies MMSE-LSA, postprocesses, saves enhanced .wav |
| `execution/05_wavelet_denoiser.py` | Loads standardized noisy .wav, applies Wavelet DWT, postprocesses, saves enhanced .wav |
| `execution/06_baseline_eval.py` | Runs both denoisers on NOIZEUS test set, computes all metrics, saves to `results/classical_baselines.csv` |

---

## Tasks (In Correct Order)

### MMSE-LSA First
- [ ] Install scipy: `pip install scipy`
- [ ] Write `execution/04_mmse_lsa_filter.py`:
  - Load a standardized noisy .wav (16kHz, mono, float32) from `data/processed/noisy/`
  - Apply `mmse_lsa_filter()`
  - Apply `postprocess_filter_output()`
  - Save enhanced .wav to `results/enhanced_mmse/`
- [ ] **Listen test:** Does it sound less noisy? Does musical noise appear? If yes, reduce gain floor.
- [ ] Run on 5 NOIZEUS clips at SNR=5dB, compute HASPI and STOI manually

### Wavelet Second
- [ ] Install PyWavelets: `pip install PyWavelets`
- [ ] Write `execution/05_wavelet_denoiser.py`:
  - Load same standardized noisy .wav files
  - Apply `wavelet_denoise()`
  - Apply `postprocess_filter_output()`
  - Save enhanced .wav to `results/enhanced_wavelet/`
- [ ] **Listen test:** Compare with MMSE-LSA output. Which sounds better for babble noise? For street noise?

### Evaluation
- [ ] Write `execution/06_baseline_eval.py`:
  - Run both denoisers on all NOIZEUS sentences (8 noise types × 4 SNR levels = 32 conditions)
  - Compute HASPI (pyclarity, audiogram=[40,40,40,40,40,40] dB HL — flat 40 dB)
  - Compute STOI (pystoi)
  - Compute PESQ (pesq library)
  - Compute SI-SDR
  - Save all results to `results/classical_baselines.csv`
- [ ] Plot bar charts of HASPI and STOI per noise type

---

## Resources

- **MMSE paper:** Ephraim & Malah (1985) — IEEE TASLP — read the abstract and gain function derivation
- **Wavelet:** PyWavelets docs — pywavelets.readthedocs.io
- **Reference implementation:** `pysepm` library has both MMSE and wavelet — use as a sanity check
- **Book:** Loizou, "Speech Enhancement: Theory and Practice" — Chapter 6 (MMSE) then Chapter 9 (Wavelet)

---

## Success Criteria

Numerical scores for both classical baselines on NOIZEUS test set:
- HASPI (with flat 40 dB HL audiogram)
- STOI
- PESQ
- SI-SDR

These numbers are the floor that every deep learning model in Phases 4–5b must beat.
MMSE-LSA should score higher than Wavelet on most noise types (it is the stronger algorithm).

---

## Learnings Log

*(Agent: append findings here — what alpha value worked, which wavelet level, any files that failed, listen-test impressions)*
