# Phase 5A Verification Report

**Project:** Audiogram-Personalized Speech Enhancement for Hearing Aids
**Scope:** Phase 5A composite-loss foundation only
**Model:** `CausalUNetCRM` class name retained for checkpoint compatibility; Phase A is currently an offline `center=True` STFT model
**Canonical training platform:** Local NVIDIA RTX 4090 workstation
**Phase B:** Not implemented in this branch and must not be launched from the Phase A trainer

---

## 1. Corrected scope

Phase 5A verifies the numerical-stability and training-pipeline changes:

- FFT-based RIR augmentation with equivalent padded `conv1d` semantics.
- Deterministic ERB filterbank loading and log-frequency audiogram interpolation.
- Suppression-only polar CRM with `|M| <= 1`.
- Stable compressed-complex and cosine phase loss terms under bf16.
- Strict VoiceBank/MUSAN/RIR file-count validation.
- 25-epoch composite-loss foundation training on the RTX 4090 (pending the
  corrected fresh workstation run).

Phase 5A does **not** establish:

- HASPI/HASQI improvement.
- Audiogram personalization benefit.
- Causal streaming equivalence.
- Final hearing-aid latency.
- Phase B HASPI surrogate or MetricGAN+/LSGAN performance.

Those claims require separate implementations and evaluation gates.

---

## 2. Corrections made to the verification workflow

The corrected evaluator in `execution/phase5_unet_film/26_eval_phase5a.py` now:

1. Uses the official non-overlapping VoiceBank-DEMAND test split:
   `clean_testset_wav/` paired with `noisy_testset_wav/`.
2. Never evaluates the dynamic training clean filelist.
3. Uses deterministic sorted pairs and records the evaluation seed.
4. Writes a manifest containing every clean/noisy test pair.
5. Records SHA-256 hashes for the checkpoint, manifest, and per-sample CSV.
6. Records Git revision, Python, PyTorch, torchaudio, CUDA, GPU, model, and
   evaluation configuration metadata.
7. Writes per-sample SI-SDR/STOI results before calculating aggregate means.
8. Records the actual profile-output difference as mean absolute waveform
   difference; it is not mislabeled as RMS.
9. Explicitly records that the Phase A model is not causal.

The old aggregate values below are retained only as historical traceability.
They were produced by the previous dynamic-mix evaluator and are **not** the
accepted final Phase 5A benchmark.

---

## 3. Historical Phase A run — not independently reproducible here

The previous report recorded:

| Metric | Historical value |
|---|---:|
| Best checkpoint epoch | 23 |
| Best validation loss | `-14.3544` |
| SI-SDR input | `6.58 dB` |
| SI-SDR enhanced | `14.72 dB` |
| SI-SDR gain | `+8.14 dB` |
| STOI input | `0.8322` |
| STOI enhanced | `0.8463` |
| Observed mask maximum | `1.0000` |

These numbers agree internally with the historical JSON artifact, but the
current Mac checkout does not contain the workstation dataset or the ignored
`.pt` checkpoint. They cannot be rerun here and must be regenerated after the
corrected Phase A retraining run.

The historical profile differences were approximately `3.75e-05` mean absolute
waveform difference for Profile A versus B. The evaluator's prior `>1e-4`
personalization threshold therefore did not pass. Phase A should be reported
as a generic denoising foundation, not as a completed personalization result.

---

## 4. Required RTX 4090 acceptance gates

Before accepting a new Phase 5A result:

```text
1. CUDA is available and the GPU is NVIDIA GeForce RTX 4090.
2. bfloat16 support is reported as true.
3. File counts are exactly clean=11071, noise=930, rir=60000.
4. The bf16 one-step gate reports finite loss, gradients, and parameters.
5. All 25 Phase A epochs complete without non-finite loss or parameters.
6. best_composite.pt and last.pt exist and are loadable.
7. The corrected evaluator processes official VoiceBank test pairs only.
8. The JSON, manifest, per-sample CSV, and checkpoint hash agree.
9. No causal or HASPI/HASQI claim is made from Phase A alone.
```

The exact setup and retraining sequence are in `phase5a_retraining_prompt.md`.

---

## 5. Current disposition

Phase 5A source safeguards are substantially corrected and ready for a fresh
RTX 4090 run. The historical Phase A result is not discarded as evidence of a
failed training attempt; it is reclassified as a legacy run whose evaluation
protocol and provenance were insufficient for final reporting.

Phase 5 overall remains active. Phase B is intentionally deferred until the
Phase A checkpoint and corrected held-out evaluation are complete.
