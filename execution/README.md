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
├── phase5_unet_film/
│   ├── 03c_data_pipeline_v2.py   — VoiceBank+MUSAN+RIR dynamic mixer and strict filelist gate
│   ├── 20_model_unet_film.py     — CausalUNetCRM + TinyUNetCRM
│   ├── 21_train_unet_film.py     — local CUDA Phase A training and checkpoint resume
│   ├── 22_streaming_inference.py — 64-hop streaming latency harness
│   └── 24_verify_local_training.py — strict local CUDA/data/one-step gate
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

# Phase 5 — run from a local RTX 4090 workstation after the strict dataset gate
python execution/phase5_unet_film/24_verify_local_training.py

python execution/phase5_unet_film/21_train_unet_film.py \
  --phase A --device cuda --mixed_precision bf16 \
  --data_dir data/processed \
  --audiograms results/phase1_audiology/audiograms.json \
  --ckpt_dir results/phase5_unet_film/checkpoints
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
| 5 | 21_train_unet_film.py | `results/phase5_unet_film/checkpoints/{last,best_composite}.pt` |
