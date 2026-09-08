# execution/ — Script Directory

All deterministic Python scripts for the project, organised by phase.
Each script is also kept in the root `execution/` folder for backward compatibility with all `python3 execution/XX_script.py` commands.

## Folder Structure

```
execution/
├── README.md                          ← this file
│
├── phase0_dsp/                        ← Phase 0: DSP Fundamentals
│   ├── 00_verify_setup.py             — import checks for all libraries
│   └── 01_stft_visualize.py           — STFT, spectrogram, HF removal demo
│
├── phase1_audiology/                  ← Phase 1: Audiology + Audiograms
│   ├── 02_audiogram_generator.py      — generates 3 test + 8 random audiograms
│   └── 03_simulate_hearing_loss.py    — MSBG hearing loss simulator (pyclarity)
│
├── phase2_data_pipeline/              ← Phase 2: Data Pipeline
│   ├── 03a_build_metadata.py          — scans datasets, builds CSV manifests
│   ├── 03b_standardize_audio.py       — resamples/normalizes/pads all audio to 16kHz 4s
│   └── 03c_data_pipeline.py           — HearingAidDataset + DataLoader test
│
└── phase3_classical_baselines/        ← Phase 3: Classical Baselines
    ├── 04_mmse_lsa_filter.py          — MMSE-LSA filter (Ephraim & Malah 1985)
    └── 05_wavelet_denoising.py        — Wavelet DWT denoising (Donoho & Johnstone 1994)
```

## Running Scripts

All scripts must be run from the **project root**, not from inside `execution/`:

```bash
# Correct:
python3 execution/04_mmse_lsa_filter.py --all

# Also correct (same file):
python3 execution/phase3_classical_baselines/04_mmse_lsa_filter.py --all
```

## Quick Reference

| Script | Usage | Output |
|--------|-------|--------|
| `00_verify_setup.py` | `python3 execution/00_verify_setup.py` | Console report |
| `01_stft_visualize.py` | `python3 execution/01_stft_visualize.py` | `results/plots/00_*.png`, `results/audio_demos/` |
| `02_audiogram_generator.py` | `python3 execution/02_audiogram_generator.py` | `results/plots/01_*.png`, `results/data/audiograms.json` |
| `03_simulate_hearing_loss.py` | `python3 execution/03_simulate_hearing_loss.py` | `results/plots/01_hearing_loss*.png`, `results/audio_demos/01_*.wav` |
| `03a_build_metadata.py` | `python3 execution/03a_build_metadata.py` | `data/metadata/*.csv` |
| `03b_standardize_audio.py` | `python3 execution/03b_standardize_audio.py` | `data/processed/**/*.wav` |
| `03c_data_pipeline.py` | `python3 execution/03c_data_pipeline.py` | `results/plots/02_dataloader_test.png` |
| `04_mmse_lsa_filter.py` | `python3 execution/04_mmse_lsa_filter.py --all` | `results/enhanced_mmse/noizeus/` |
| `05_wavelet_denoising.py` | `python3 execution/05_wavelet_denoising.py --all` | `results/enhanced_wavelet/noizeus/` |
