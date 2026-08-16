# Directive 02 — Classical Baselines (Wavelet Denoising + MMSE-LSA Filter)

**Phase:** 2  
**Goal:** Implement two classical (non-deep-learning) speech enhancement baselines. These are the reference points every deep learning model must beat.  
**Estimated time:** 1 week (Aug 30 – Sep 6, 2026)  
**Lead:** Jwanil  
**Output:** Two working classical denoisers with documented metric scores on NOIZEUS.

---

## Why Two Classical Baselines (Not One)?

The faculty recommended replacing the basic Wiener filter with more sophisticated classical approaches. We now have:

| Baseline | Technique | Why It's Included |
|---|---|---|
| **Wavelet Denoising** | Discrete Wavelet Transform (DWT) + soft thresholding | Best classical approach for non-stationary noise. Adaptive time-frequency resolution. Used in real hearing aids. |
| **MMSE-LSA Filter** | Minimum Mean Square Error – Log Spectral Amplitude | The industry standard in commercial hearing aids today. Beating it is a meaningful claim. |

Having two classical baselines is stronger than one — it shows you understand the landscape and gives the deep learning models two real competitors, not one toy.

---

## Background Reading

### Discrete Wavelet Transform (DWT) Denoising
The DWT converts a 1D audio signal into **wavelet coefficients** at multiple scales (like a spectrogram, but with adaptive resolution). The key insight: speech has energy at specific scales, noise is spread everywhere. By thresholding (zeroing out) small coefficients, you remove noise while preserving speech structure.

**Algorithm (Donoho & Johnstone, 1994):**
```
1. Compute DWT(signal) → wavelet coefficients at levels 1..L
2. Estimate noise standard deviation:
   σ = median(|coefficients at finest level|) / 0.6745
3. Compute universal threshold:
   λ = σ × √(2 × log(N))    where N = signal length
4. Apply soft threshold to detail coefficients:
   coeff_new = sign(coeff) × max(|coeff| - λ, 0)
5. Keep approximation coefficients untouched
6. IDWT(thresholded coefficients) → denoised signal
```

**Wavelet choice:** `db8` (Daubechies-8) or `sym8` (Symlet-8) — standard for speech signals. Decomposition level: 5–6 at 16 kHz, covering the full speech frequency range.

### MMSE-LSA (Minimum Mean Square Error – Log Spectral Amplitude) Filter
Introduced by Ephraim & Malah (1985). The industry standard in hearing aids. Key improvements over the basic Wiener filter:

- Uses a **decision-directed approach** to estimate the a priori (prior) Signal-to-Noise Ratio (SNR), which smooths gain estimates over time
- Operates on the **log spectral amplitude** rather than linear amplitude, which better matches human auditory perception
- Dramatically reduces "musical noise" — the annoying tonal artifacts that plague simpler approaches

**The decision-directed SNR estimate (updated every frame):**
```
ξ(k,n) = α × [Â²(k,n-1) / λ_n(k,n)] + (1-α) × max(γ(k,n) - 1, 0)

where:
  ξ  = a priori SNR estimate for frequency bin k at frame n
  α  = 0.98 (smoothing factor — most weight on previous estimate)
  Â  = previous enhanced spectral amplitude
  λ_n = noise power estimate (updated using minimum statistics)
  γ  = instantaneous SNR = |Y(k,n)|² / λ_n(k,n)
```

---

## Execution Scripts

| Script | What It Does |
|--------|-------------|
| `execution/04_wavelet_denoiser.py` | Loads a noisy .wav file, applies DWT soft thresholding, saves enhanced .wav |
| `execution/05_mmse_lsa_filter.py` | Loads a noisy .wav file, applies MMSE-LSA decision-directed filtering, saves enhanced .wav |
| `execution/06_baseline_eval.py` | Runs both denoisers on the NOIZEUS test set, computes HASPI, HASQI, STOI, SI-SDR, PESQ, and saves results to `results/classical_baselines.csv` |

---

## Tasks

- [ ] Install PyWavelets: `pip install PyWavelets`
- [ ] Write `execution/04_wavelet_denoiser.py`:
  - Load noisy .wav with torchaudio
  - Apply DWT using `pywt.wavedec()` with wavelet='db8', level=5
  - Estimate noise sigma via Median Absolute Deviation (MAD) on finest detail coefficients
  - Compute universal threshold λ = σ × √(2 × log(N))
  - Apply `pywt.threshold()` with mode='soft' to all detail coefficients
  - Reconstruct with `pywt.waverec()`
  - Save output .wav
- [ ] Write `execution/05_mmse_lsa_filter.py`:
  - Implement decision-directed a priori SNR estimation (α = 0.98)
  - Implement noise floor tracking (minimum statistics, 1.5s window)
  - Compute MMSE-LSA gain function
  - Apply gain to spectrogram, reconstruct with ISTFT
- [ ] Write `execution/06_baseline_eval.py`:
  - Run both denoisers on all NOIZEUS sentences (all 8 noise types, all 4 SNR levels)
  - Compute HASPI and HASQI (use a synthetic audiogram for evaluation: flat 40 dB HL)
  - Compute STOI using pystoi
  - Compute SI-SDR
  - Save to CSV
- [ ] **Listen test:** Load a babble-noise clip. Run both denoisers. Listen. Can you hear the musical noise reduced in MMSE-LSA vs wavelet? If they both sound terrible, debug.

---

## Resources

- **DWT:** PyWavelets docs — https://pywavelets.readthedocs.io/en/latest/
- **MMSE reference:** Ephraim & Malah (1985) — "Speech Enhancement Using a Minimum Mean Square Error Log-Spectral Amplitude Estimator", IEEE TASLP
- **Reference implementation:** `pysepm` library has MMSE implementations
- **Book:** Loizou, "Speech Enhancement: Theory and Practice", Chapter 6 (MMSE) and Chapter 9 (Wavelet)
- **Video:** YouTube — search "Wavelet denoising explained" (~15 min, several good options)

---

## Success Criteria

You have numerical scores for both classical baselines on the NOIZEUS test set:
- HASPI (with flat 40 dB HL audiogram)
- STOI
- SI-SDR

These numbers are the floor that every deep learning model in Phases 4-5b must beat.

---

## Learnings Log

*(Agent: append findings here — note which wavelet level works best, what α value you settled on, any dataset quirks)*
