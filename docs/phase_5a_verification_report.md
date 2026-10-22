# Phase 5A Verification Report

**Scope:** Phase 5A composite-loss foundation only
**Model:** `CausalUNetCRM` class name retained for checkpoint compatibility
**Phase B:** Not implemented in this branch

## 1. Reproducibility record

- Git commit: `f548253eb66ab4e285e199d8af9af68aee2164e1`
- Git tracked tree clean: `True`
- Checkpoint: `results\phase5_unet_film\checkpoints\best_composite.pt`
- Checkpoint SHA-256: `94de1a1ef6ef34af3e136a3899d12ec126ef85c01c44c6780603b59f2bfb0696`
- Checkpoint epoch: `22`
- Checkpoint validation loss: `-14.484773329326085`
- Evaluation protocol: `official_voicebank_test_pairs`
- Evaluation pairs: `824`
- Manifest: `results\phase5_unet_film\phase5a_eval_manifest.csv`
- Manifest SHA-256: `98216ac42a29f22965d6ef15028cb6222cb65be39bf462d53da421c8fb0c3b85`
- Per-sample CSV: `results\phase5_unet_film\phase5a_eval_per_sample.csv`
- Per-sample CSV SHA-256: `8364b3e139689b08e721fb8fcd6daf0cf9b676c71e7a694b7d27879422886070`
- Evaluator command: `["C:\\Users\\ADMIN\\.gemini\\antigravity-ide\\scratch\\hearing-aid-speech-enhancement\\.venv\\Scripts\\python.exe", "C:\\Users\\ADMIN\\.gemini\\antigravity-ide\\scratch\\hearing-aid-speech-enhancement\\execution\\phase5_unet_film\\26_eval_phase5a.py", "--data_dir", "data/processed", "--checkpoint", "results/phase5_unet_film/checkpoints/best_composite.pt", "--results_json", "results/phase5_unet_film/phase5a_eval_results.json", "--results_csv", "results/phase5_unet_film/phase5a_eval_per_sample.csv", "--manifest", "results/phase5_unet_film/phase5a_eval_manifest.csv", "--seed", "0"]`
- Seed: `0`
- PyTorch: `2.6.0+cu124`; torchaudio: `2.6.0+cu124`
- CUDA: `12.4`; GPU: `NVIDIA GeForce RTX 4090`

## 2. Held-out benchmark

The evaluator uses the official non-overlapping VoiceBank-DEMAND `clean_testset_wav`/`noisy_testset_wav` pairs. It does not call the dynamic training loader and does not add random noise, RIR, crop, or SNR augmentation.

| Profile | SI-SDR input | SI-SDR enhanced | SI-SDR gain | STOI input | STOI enhanced | STOI gain |
|---|---:|---:|---:|---:|---:|---:|
| Profile_A_Normal | 8.45 | 13.80 | 5.35 | 0.9203 | 0.9188 | -0.0014 |
| Profile_B_Presbycusis | 8.45 | 13.81 | 5.36 | 0.9203 | 0.9188 | -0.0014 |
| Profile_C_SevereFlat | 8.45 | 13.81 | 5.36 | 0.9203 | 0.9188 | -0.0015 |

## 3. Safety and personalization observations

- Maximum observed mask magnitude: `1.000000`.
- Suppression-only mask invariant: `True`.
- Mean absolute output difference, Profile A vs B: `6.5436676e-05`.
- Mean absolute output difference, Profile B vs C: `3.040702e-05`.
- Personalization signal threshold (`>1e-4`) passed: `False`.

Phase A is an offline foundation model using `center=True` STFT/iSTFT. This report makes no causal streaming latency claim and reports no HASPI/HASQI result. Those are separate Phase B and deployment evaluation gates.

## 4. Acceptance status

This report is valid only when the checkpoint, manifest, per-sample CSV, and all recorded hashes remain available together. Phase B must not be started from this trainer until its HASPI surrogate and discriminator are implemented and independently validated.
