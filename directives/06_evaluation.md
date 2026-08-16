# Directive 06 — Full Evaluation

**Phase:** 6  
**Goal:** Rigorously compare all 5 models on the same test set using hearing-aid-specific metrics, and document the latency and model size of every model honestly.  
**Estimated time:** 3 days (Oct 26 – Oct 28, 2026)  
**Lead:** Both  
**Output:** A results table with hard numbers for all models, plots, and audio demonstrations.

---

## The Full Model Lineup

By the time you reach this phase, you should have trained all 5 models:

| # | Model Name | Type | Key Feature |
|---|---|---|---|
| 1 | Wavelet Denoising | Classical (no learning) | DWT soft thresholding |
| 2 | MMSE-LSA Filter | Classical (no learning) | Minimum Mean Square Error – Log Spectral Amplitude decision-directed filter |
| 3 | 1D CNN (Conv-TasNet) | Deep learning — no audiogram | Temporal Convolutional Network on raw waveform |
| 4 | U-Net + Attention + FiLM | Deep learning — personalised | Complex masking + self-attention + FiLM audiogram conditioning |
| 5 | Mamba + FiLM | Deep learning — personalised | State Space Model + FiLM audiogram conditioning — current SOTA approach |

---

## Metrics to Compute

Compute all of these for every model on every test condition:

| Metric | Full Name | Library | Notes |
|---|---|---|---|
| **HASPI** | Hearing Aid Speech Perception Index | `pyclarity` | Primary metric. Takes audiogram as input. Run for each of the 3 test audiograms. |
| **HASQI** | Hearing Aid Speech Quality Index | `pyclarity` | Quality counterpart to HASPI. Same 3 audiograms. |
| **STOI** | Short-Time Objective Intelligibility | `pystoi` | Standard reference — does NOT use audiogram. Compare to HASPI to verify personalisation effect. |
| **SI-SDR** | Scale-Invariant Signal-to-Distortion Ratio | `asteroid.losses` | Signal quality, in dB. |
| **PESQ** | Perceptual Evaluation of Speech Quality | `pesq` (`pip install pesq`) | For comparison with SE literature benchmarks. |
| **Latency** | Processing delay per 10ms frame | `torch.profiler` | Must be under 10ms for real-time hearing aid use. |
| **Parameters** | Total number of trained weights | `sum(p.numel() for p in model.parameters())` | Model size proxy. |

### The 3 Test Audiogram Profiles

Evaluate each DL model with these 3 audiograms to show personalisation effect:

```python
audiogram_A = [10, 10, 10, 10, 10, 10]    # "Normal" — flat, all normal
audiogram_B = [10, 15, 20, 45, 70, 85]    # "HF loss" — severe high-frequency loss (most common)
audiogram_C = [60, 60, 60, 60, 60, 60]    # "Flat severe" — uniform severe loss
```

Classical models (Wavelet, MMSE-LSA) have no audiogram input — evaluate them once on audiogram B as their baseline score.

---

## Evaluation Strategy

### HASPI Expected Pattern

If personalisation is working, the HASPI improvement of Model 4 and 5 over Model 3 should be:
- **Largest for audiogram B (HF loss):** The audiogram has a very clear pattern — the model should learn to specifically preserve high-frequency speech content
- **Moderate for audiogram C (flat severe):** Harder, uniform loss
- **Smallest for audiogram A (normal):** Not much personalisation needed

If HASPI improvement is the same for all audiograms → the model is not using the audiogram effectively.

### STOI vs HASPI Comparison

The key scientific claim of the project:

```
Expected result for Models 4 & 5 vs Model 3:
  ΔHASPI (improvement) >> ΔSTOI (improvement)

This means: the personalised model helps hearing-impaired listeners MORE than
it helps general listeners. The improvement is specifically due to audiogram
conditioning, not just better noise removal.
```

If ΔHASPI ≈ ΔSTOI, the FiLM conditioning is not providing hearing-specific benefit.

---

## Execution Scripts

| Script | What It Does |
|--------|-------------|
| `execution/23_evaluate_all_models.py` | Runs all 5 models on NOIZEUS test set, computes all metrics, saves to `results/full_results.csv` |
| `execution/24_latency_benchmark.py` | Measures inference latency per 10ms frame for each model on CPU (and GPU if available) |
| `execution/25_plot_results.py` | Generates: (1) bar chart comparing all models on HASPI, (2) HASPI vs audiogram profile plot, (3) accuracy vs latency scatter plot |
| `execution/26_generate_audio_demos.py` | Saves before/after .wav files for 3 noise conditions × 3 audiograms × 5 models |

---

## Tasks

- [ ] Write `execution/23_evaluate_all_models.py`:
  - Load each model from its saved checkpoint
  - Run all NOIZEUS sentences through all models
  - For each DL model, run with all 3 audiogram profiles
  - Compute: HASPI (pyclarity), HASQI (pyclarity), STOI (pystoi), SI-SDR (asteroid), PESQ (pesq library)
  - Save full results to `results/full_results.csv` with columns: `[model, noise_type, snr, audiogram_profile, haspi, hasqi, stoi, si_sdr, pesq]`

- [ ] Write `execution/24_latency_benchmark.py`:
  - Use `torch.profiler` or simply `time.time()` around a forward pass
  - Simulate a real-time scenario: feed 10ms audio frames one at a time
  - Report mean ± standard deviation latency over 1000 frames
  - Report total parameter count for each model
  - Save to `results/latency_benchmark.csv`

- [ ] Write `execution/25_plot_results.py`:
  - **Plot 1:** Bar chart — HASPI for all 5 models (for audiogram B — HF loss)
  - **Plot 2:** Grouped bar chart — HASPI for Models 3/4/5 across all 3 audiogram profiles (shows personalisation effect)
  - **Plot 3:** Scatter plot — latency (x-axis, ms) vs HASPI (y-axis) — shows the accuracy-speed trade-off
  - **Plot 4:** HASPI vs STOI improvement (Models 4 and 5 over Model 3) — should show HASPI improvement > STOI improvement
  - Save all plots to `results/plots/`

- [ ] Write `execution/26_generate_audio_demos.py`:
  - Choose 3 representative test sentences (one per noise type: babble, car, restaurant)
  - For each: save `noisy.wav`, `model1_wavelet.wav`, `model2_mmse.wav`, `model3_cnn.wav`, `model4_unet.wav`, `model5_mamba.wav`
  - For Models 4 and 5, generate one version per audiogram profile
  - Save to `results/audio_demos/`
  - Normalise all files to the same loudness level for fair listening

- [ ] **The listening test:** Listen to all audio demos. Your subjective impression should align with the numbers. If something sounds wrong, investigate.

- [ ] Write the results section of the report using these numbers.

---

## Resources

- **pyclarity HASPI/HASQI:** `pip install pyclarity` — see their documentation for API
- **pystoi STOI:** `pip install pystoi` — very simple API: `stoi(ref, deg, fs, extended=False)`
- **pesq:** `pip install pesq` — `pesq(fs, ref, deg, mode='wb')` (mode='wb' for wideband 16kHz)
- **torch.profiler:** PyTorch documentation on profiling — for accurate latency measurement
- **matplotlib:** For all plots — see matplotlib gallery for bar chart and scatter plot examples

---

## Success Criteria

You have:
1. A fully populated `results/full_results.csv` with all 5 models × all metrics × all audiograms × all NOIZEUS test conditions
2. A `results/latency_benchmark.csv` with latency and parameter count for all 5 models
3. 4 publication-quality plots saved to `results/plots/`
4. Audio demos in `results/audio_demos/` that clearly demonstrate improvement
5. HASPI improvement of Models 4/5 over Model 3 is larger than STOI improvement (the key scientific result)
6. You can state honestly which models meet the <10ms hearing aid latency constraint and which do not

---

## Learnings Log

*(Agent: append findings here — any surprising results, models that underperformed expectations, evaluation gotchas)*
