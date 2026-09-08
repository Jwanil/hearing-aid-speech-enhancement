# Execution Scripts

All scripts are organised by project phase. Each subfolder is self-contained
for that phase. Run scripts from the project root.

## Structure

```
execution/
├── phase0_dsp/
│   ├── 00_verify_setup.py        — verify environment + dependencies
│   └── 01_stft_visualize.py      — STFT/ISTFT pipeline + spectrogram plots
│
├── phase1_audiology/
│   ├── 02_audiogram_generator.py — synthesise clinical audiogram profiles
│   └── 03_simulate_hearing_loss.py — MSBG hearing loss simulation
│
├── phase2_data_pipeline/
│   ├── 03a_build_metadata.py     — scan NOIZEUS/MUSAN/TIMIT, write metadata CSV
│   ├── 03b_standardize_audio.py  — resample → 16kHz, mono, 4s, peak-normalise
│   └── 03c_data_pipeline.py      — end-to-end pipeline runner (calls 03a + 03b)
│
├── phase3_classical_baselines/
│   ├── 04_mmse_lsa_filter.py     — MMSE-LSA filter (Ephraim & Malah 1985), all 930 files
│   ├── 05_wavelet_denoising.py   — Wavelet DWT denoising (Donoho & Johnstone 1994)
│   └── 06_baseline_eval.py       — STOI/PESQ/SI-SDR evaluation for all baselines
│
└── README.md                     — this file
```

## How to Run

All scripts use `uv` for dependency management. Run from project root:

```bash
# Phase 0
uv run --python 3.14 python execution/phase0_dsp/00_verify_setup.py

# Phase 3 — evaluation (needs SSD mounted)
uv run --with pesq --with pystoi --with soundfile --with matplotlib \
       --python 3.14 python execution/phase3_classical_baselines/06_baseline_eval.py
```

## Outputs

| Phase | Script | Key Output |
|-------|--------|------------|
| 0 | 01_stft_visualize.py | `results/plots/phase0_dsp/*.png` |
| 1 | 02_audiogram_generator.py | `results/plots/phase1_audiology/*.png`, `results/audio_demos/phase1_audiology/*.wav` |
| 1 | 03_simulate_hearing_loss.py | `results/audio_demos/phase1_audiology/*.wav` |
| 2 | 03c_data_pipeline.py | `data/processed/` (standardised WAVs on SSD) |
| 3 | 04_mmse_lsa_filter.py | `results/enhanced_mmse/noizeus/` |
| 3 | 05_wavelet_denoising.py | `results/enhanced_wavelet/noizeus/` |
| 3 | 06_baseline_eval.py | `results/classical_baselines.csv`, `results/plots/phase3_baselines/` |
