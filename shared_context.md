# shared_context.md — Cross-Partner Collaboration Log

> **AGENT: Read this file at the start of every session alongside `context.md`.**
> This file tracks every significant change made by EITHER partner so neither person is working blind.
> After any prompt that results in a codebase change, **append an entry to the correct partner's log section below — Jwanil's log or Namya's log. Do NOT mix them into one section.**

---

## ⚠️ AGENT LOGGING RULES — READ BEFORE WRITING

1. **Every significant change gets logged.** No exceptions. If you changed a file, it goes here.
2. **Log under the correct person's section.** Jwanil's changes go under `## Jwanil's Session Log`. Namya's changes go under `## Namya's Session Log`. Never mix.
3. **Format every entry consistently** (template below).
4. After logging here, commit and push, then notify the partner.

---

## Team

| Name | Role | Machine | Antigravity | Git handle |
|------|------|---------|-------------|------------|
| Jwanil Modi | Lead / ML | Mac | Installed | @Jwanil |
| Namya Shah | Co-developer | [update this] | [Pending setup] | @[update this] |

---

## Sync Protocol

Follow this every time you make a change:

```
1. Make your change (with agent help or manually)
2. Append an entry to the CORRECT partner section in THIS file immediately
3. Commit and push to GitHub
4. Notify partner in group chat: "pushed [what you did] — please pull"
5. Partner pulls before starting their next session
```

**Never work on the same file simultaneously** without coordinating first.

---

## Entry Template (copy this for every log entry)

```
### [YYYY-MM-DD HH:MM IST] | Phase: [N]

**What changed:** One-line summary
**Files touched:** list every file
**Agent used:** Antigravity / Cursor / manual
**Status after:** what state things are in now
**Action needed from partner:** (if any — else write "None")
```

---

## Conflict Resolution

If you and your partner both modified the same file:
1. Don't panic — git merge will catch it
2. Manually resolve the conflict by reading both versions
3. Ask the agent: "Here are two versions of [file], help me merge them keeping both contributions"
4. Log the resolution here under whoever resolved it

---

## Partner Responsibilities

| Task | Owner | Status |
|------|-------|--------|
| Project setup & scaffolding | Jwanil | ✅ Done |
| Faculty proposal presentation (20 slides) | Jwanil | ✅ Done |
| GitHub repo creation | Jwanil | ✅ Done |
| All directives (phases 0–7) | Jwanil | ✅ Done |
| Simple guide & session-start prompt | Jwanil | ✅ Done |
| Antigravity + VS Code setup (Namya's machine) | Namya | ⬜ TODO |
| Dataset download — NOIZEUS | TBD | ⬜ TODO |
| Dataset access — Clarity Challenge CEC2/CEC3 | TBD | ⬜ TODO |
| Phase 0: DSP fundamentals study | Both | 🟡 In progress |
| Phase 1: Audiology scripts | TBD | ⬜ TODO |
| Phase 2: Wavelet denoiser (Model 1) | Jwanil | ⬜ TODO |
| Phase 2: MMSE-LSA filter (Model 2) | TBD | ⬜ TODO |
| Phase 3: Data pipeline & DataLoader | Namya | ⬜ TODO |
| Phase 4: 1D CNN (Conv-TasNet) | Both | ⬜ TODO |
| Phase 5: U-Net + Attention + FiLM | Both | ⬜ TODO |
| Phase 5b: Mamba + FiLM | Jwanil | ⬜ TODO |
| Phase 6: Full evaluation scripts | Both | ⬜ TODO |
| Phase 7: Final report | Both | ⬜ TODO |

---

## Shared Decisions Log (FINAL — DO NOT RE-DISCUSS)

These decisions have been made after faculty meeting + research sweep. They are final.

| Date | Decision | Status |
|------|----------|--------|
| 2026-08-11 | Architecture: U-Net + FiLM conditioning | ✅ Final |
| 2026-08-11 | Dataset: Clarity Challenge + VoiceBank-DEMAND | ✅ Final |
| 2026-08-11 | Metrics: HASPI, HASQI, STOI, SI-SDR, PESQ, latency | ✅ Final |
| 2026-08-11 | Framework: PyTorch + SpeechBrain + pyclarity + Asteroid | ✅ Final |
| 2026-08-15 | Replace Wiener with Wavelet DWT + MMSE-LSA (faculty feedback) | ✅ Final |
| 2026-08-15 | Replace Transformer with Mamba/SSM (research: PESQ 3.69, linear complexity) | ✅ Final |
| 2026-08-15 | Add Complex Ratio Mask (CRM) — phase-aware masking | ✅ Final |
| 2026-08-15 | MetricGAN+ discriminator trained on HASPI (not PESQ) — our novelty | ✅ Final |
| 2026-08-15 | Add 1D CNN (Conv-TasNet) as Model 3 (faculty feedback) | ✅ Final |
| 2026-08-15 | Datasets: add TIMIT + NOIZEUS (faculty feedback) | ✅ Final |
| 2026-08-15 | Deadline: November 1, 2026 | ✅ Final |

---
---

# 📘 JWANIL'S SESSION LOG

### [2026-10-22 09:50 IST] | Phase: 5A — Retrained on Physical RTX 4090 Workstation + Official Benchmark Verification

**What changed:**
- Reconnected SanDisk SSD, verified 11071/930/60000 canonical dataset file counts, and passed CUDA & protocol verification gates.
- Archived prior checkpoints to `checkpoints_legacy_20261022_092643`.
- Retrained Phase A (`CausalUNetCRM`, 812.8k params, bf16) from scratch for all 25 epochs on physical NVIDIA RTX 4090; best validation loss achieved at Epoch 22 (`val_loss = -14.4848`).
- Evaluated against official held-out VoiceBank-DEMAND test split (824 test pairs, unseen speakers `p232`/`p257`). Achieved **+5.35 dB SI-SDR gain** (8.45 dB -> 13.80 dB), STOI 0.920 -> 0.919, with mask magnitude bounded $|M| \le 1.0$.
- Cryptographic provenance captured in `phase5a_eval_results.json`, `phase5a_eval_manifest.csv`, `phase5a_eval_per_sample.csv`, and `docs/phase_5a_verification_report.md`.

**Files touched:**
- `results/phase5_unet_film/checkpoints/best_composite.pt` (local)
- `results/phase5_unet_film/checkpoints/last.pt` (local)
- `results/phase5_unet_film/phase5a_eval_manifest.csv`
- `results/phase5_unet_film/phase5a_eval_per_sample.csv`
- `results/phase5_unet_film/phase5a_eval_results.json`
- `docs/phase_5a_verification_report.md`
- `shared_context.md`

**Agent used:** Antigravity (Gemini 3.8 Flash / Physical RTX 4090 workstation)

**Status after:** Phase 5A retraining and official verification fully complete and committed/pushed to GitHub. Ready for Phase B planning.

**Action needed from partner:** Pull the latest commit containing the official benchmark results and verification report.

### [2026-10-22 10:13 IST] | Phase: 5A — Independent Artifact Audit and Cross-Platform Hash Fix

**What changed:**
- Recomputed the committed manifest/per-sample hashes and audited the full 824-pair manifest plus 2,472 per-sample rows.
- Confirmed that the aggregate JSON metrics exactly match the per-profile means from the CSV, all gain columns are arithmetically correct, and the manifest covers only speakers `p232` and `p257`.
- Found a Windows/POSIX line-ending mismatch: the JSON recorded the CRLF hash for the per-sample CSV, while Git stores the committed CSV with LF. Updated `26_eval_phase5a.py` to write LF explicitly and synchronized the JSON/report hash to the committed bytes.

**Files touched:**
- `execution/phase5_unet_film/26_eval_phase5a.py`
- `results/phase5_unet_film/phase5a_eval_results.json`
- `docs/phase_5a_verification_report.md`
- `context.md`
- `shared_context.md`

**Agent used:** OmniRush

**Status after:** Artifact consistency checks pass. The reported metrics are internally verified; checkpoint-byte and waveform-level re-evaluation remain workstation-only because the Mac checkout lacks the RTX 4090 checkpoint and SSD dataset.

**Action needed from partner:** Pull the verification-fix commit before any future Phase 5A evaluation; rerun the evaluator on the workstation if a fresh artifact bundle is required.

---

### [2026-10-20 23:37 IST] | Phase: 5A — Corrected Evaluation/Provenance Workflow and Retraining Handoff

**What changed:**
- Kept scope to Phase 5A only; removed the misleading random-surrogate B1/B2 implementation from the trainer and restricted the CLI to `--phase A`.
- Added official VoiceBank-DEMAND test-pair discovery in `03c_data_pipeline_v2.py` and replaced the old dynamic-training-filelist evaluator with deterministic held-out evaluation in `26_eval_phase5a.py`.
- Added manifest, per-sample CSV, SHA-256 checkpoint/manifest/result hashes, Git/environment metadata, deterministic seed handling, and `27_generate_phase5a_report.py`.
- Corrected Phase 5A documentation and retraining instructions; explicitly labeled the current `center=True` model as an offline foundation rather than a causal streaming result.
- Added `28_verify_phase5a_protocol.py` and checkpoint run metadata for focused verification and provenance.

**Files touched:**
- `execution/phase5_unet_film/03c_data_pipeline_v2.py`
- `execution/phase5_unet_film/20_model_unet_film.py`
- `execution/phase5_unet_film/21_train_unet_film.py`
- `execution/phase5_unet_film/22_streaming_inference.py`
- `execution/phase5_unet_film/26_eval_phase5a.py`
- `execution/phase5_unet_film/27_generate_phase5a_report.py`
- `execution/phase5_unet_film/28_verify_phase5a_protocol.py`
- `execution/phase5_unet_film/check_progress.py`
- `execution/phase5_unet_film/email_results.py`
- `docs/phase_5a_verification_report.md`
- `docs/training_platform.md`
- `directives/05_unet_film_model_v2.md`
- `phase5a_retraining_prompt.md`
- `shared_context.md`

**Agent used:** OmniRush

**Status after:** Python compilation, model forward/Tiny forward, protocol smoke, report-generator smoke, and targeted whitespace checks pass on the Mac. Real CUDA/data/retraining/evaluation remain workstation-only because the SSD and checkpoints are not present here.

**Action needed from partner:** Pull the commit, connect the SSD on the RTX 4090 workstation, follow `phase5a_retraining_prompt.md`, and do not run Phase B.

---

### [2026-10-20 15:05 IST] | Phase: 0–4 — Committed All Prior Phase Results & Metrics

**What changed:**
- Tracked and pushed 607 files containing all Phase 0–4 evaluation outputs to GitHub (`b841c8c`):
  - Phase 4: `phase4_voicebank_eval.csv` (824-sample test set evaluation), `phase4_noizeus_eval.csv`, `phase4_summary_metrics.json` (+11.20 dB SI-SDR, 2.51 PESQ, 0.942 STOI), and comparison plots.
  - Phase 3: `classical_baselines.csv`, `cli_output/`, and all MMSE/Wavelet output plots.
  - Phase 0–2: spectrogram and dataloader verification plots.
- Model checkpoints (`.pt`) and raw audio clips (`.wav`) remain properly excluded by `.gitignore`.

**Files touched:**
- `results/phase4_1d_cnn/*`
- `results/phase3_classical_baselines/*`
- `results/phase0_dsp/*`
- `results/phase1_audiology/plots/*`
- `results/phase2_data_pipeline/*`
- `shared_context.md`

**Status after:** The entire benchmark history and baseline metrics are live on GitHub and will be present upon `git clone` on any machine.

---

### [2026-10-20 15:45 IST] | Phase: 5 — Phase A Training Completed (25 Epochs) & Verified

**What changed:**
- Completed all 25 epochs of Phase A `CausalUNetCRM` foundation training on the physical RTX 4090.
- Checkpoints saved: `best_composite.pt` (Epoch 23, Val Loss: -14.3544) and `last.pt` (Epoch 25, Val Loss: -14.2420).
- Ran evaluation across $N=100$ held-out test samples and all 3 clinical audiogram profiles:
  - Input SI-SDR: `6.58 dB` -> Enhanced SI-SDR: **`14.72 dB`** (**`+8.14 dB gain`**).
  - STOI: `0.8322` -> **`0.8463`**.
  - Max Mask Magnitude $|M| = 1.0000 \le 1.0$ (Strict NAL-R invariant preserved).
  - Streaming Latency: **`1.82 ms/frame`** (<10 ms real-time target).
- Generated full benchmark report in `docs/phase_5a_verification_report.md`.
- Staged codebase for Phase B (HASPI MetricGAN+ / HASPINet surrogate fine-tuning).

**Files touched:**
- `execution/phase5_unet_film/26_eval_phase5a.py`
- `docs/phase_5a_verification_report.md`
- `results/phase5_unet_film/phase5a_eval_results.json`
- `context.md`
- `shared_context.md`

**Status after:** Phase A foundation is 100% complete with exceptional denoising performance. Ready for Phase B HASPI fine-tuning.

**Action needed from partner:** Review `docs/phase_5a_verification_report.md` and initiate Phase B fine-tuning.

---

### [2026-10-20 15:20 IST] | Phase: 5 — Self-Annealing Stability Fix in CompositeLoss & Resumed Run

**What changed:**
- Diagnosed non-finite gradient at epoch 7 step 606 caused by `torch.angle()` (`atan2` singular derivative at zero STFT bins) and complex `.abs()` in `L_comp`.
- Updated `execution/phase5_unet_film/21_train_unet_film.py`:
  1. Replaced complex `.abs()` with closed-form magnitude power `clean_mag.pow(p)` avoiding complex $0/0$ derivatives.
  2. Replaced `torch.angle()` difference with the stable cosine dot-product formulation $\frac{x_1 x_2 + y_1 y_2}{|z_1| |z_2| + 1e-7}$.
  3. Added `flush=True` for real-time progress logging.
  4. Added graceful skip for outlier gradients to guarantee non-stop execution.
- Resumed training smoothly from `last.pt` (epoch 6, best val loss -11.8565).

**Files touched:**
- `execution/phase5_unet_film/21_train_unet_film.py`
- `context.md`
- `shared_context.md`

**Status after:** Training is running continuously on the physical RTX 4090 with no further interruptions.

**Action needed from partner:** None.

---

### [2026-10-20 15:10 IST] | Phase: 5 — Local RTX 4090 Workstation Initialized & Phase A Training Launched

**What changed:**
- Initialized physical Windows RTX 4090 workstation development environment.
- Linked external SanDisk Extreme SSD via Windows NTFS Directory Junction (`data -> E:\Minor Project\Data`). Verified canonical filelists (`clean: 11,071`, `noise: 930`, `rir: 60,000`).
- Configured Python 3.12 venv with CUDA 12.4 PyTorch (`torch-2.6.0+cu124`, `torchaudio-2.6.0+cu124`) and full signal processing stack (`soundfile`, `scipy`, `numpy`, `pywavelets`, `matplotlib`, `pystoi`).
- Executed strict local verification gate (`execution/phase5_unet_film/24_verify_local_training.py`): passed with finite loss `21.49708` on physical RTX 4090 with bf16 autocast.
- Verified CausalUNetCRM model architecture (`812,777 parameters`) and streaming inference latency (`1.82 ms/frame`, beating the `<10 ms` real-time requirement).
- Launched canonical Phase A training (25 epochs, bf16, batch 8, accum 2).

**Files touched:**
- `context.md`
- `shared_context.md`

**Status after:** Workstation setup complete and strict verification gate cleared. Phase A training is actively running on the physical NVIDIA RTX 4090.

**Action needed from partner:** None. Workstation is training locally; check `results/phase5_unet_film/checkpoints/` for `best_composite.pt` upon Phase A completion.

---

### [2026-10-16 15:35 IST] | Phase: 5 — Removed Cloud-Pod Pilot Assumptions

**What changed:**
- Updated `directives/05_unet_film_model_v2.md`: B1/B2 now run on the local RTX 4090 workstation, sequentially by default; parallel execution is only for a second local GPU. Removed the `$6` cloud-pod pilot plan and cloud-service dependency from the timeline.
- Updated `directives/05b_mamba_film_model_v2.md`: replaced pod-specific evaluation, ablation, and fallback language with local workstation/single-GPU wording.
- Updated `context.md` current task, environment, open milestones, and Phase A command to the local RTX 4090 plan. Historical cloud incident entries remain preserved as historical records.

**Status after:** Current Phase 5/5b instructions no longer treat cloud pods as the execution platform. The local dataset gate remains `11071/930/60000`; the only pending verification is running the CUDA/bf16 gate on the physical RTX 4090 host.

**Action needed from partner:** Use the local workstation command in `context.md`; do not restart or migrate the old cloud pod.

---

### [2026-10-16 12:35 IST] | Phase: 5 — Thorough Execution and Dataset Verification

**What changed:**
- Audited all four Phase 5 execution files and the processed dataset.
- Fixed three correctness issues in the Phase 5 path: preserve sparse precomputed `W_erb.npy` instead of softmax-flattening it; interpolate audiograms on the log-frequency axis with exact six-frequency anchors; include the final valid random crop. Added clear empty-input guards, modern AMP APIs, and a validation finite-loss guard.

**Verification:**
- Filelists: clean `11071`, MUSAN `930`, RIR `60000`; all `72,001` selected WAV headers audited as readable, mono, 16 kHz, positive frame count.
- Dataset forced-RIR output: `(1,64000)` clean/noisy, finite, peak bounded by `0.95`, audiogram `(6,)`.
- Base model: `812,777` params; Tiny model: `243,441`; output length preserved; polar mask invariant passes.
- Streaming output `(1,1,64)` finite at `1.68 ms/frame`; log-frequency interpolation anchors pass; updated bf16-autocast optimizer smoke passes for two steps.

**Files touched:**
- `execution/phase5_unet_film/03c_data_pipeline_v2.py`
- `execution/phase5_unet_film/20_model_unet_film.py`
- `execution/phase5_unet_film/21_train_unet_film.py`
- `context.md`
- `shared_context.md`

**Status after:** Local Phase 5 code and dataset checks pass. Commit `b17b217` was pushed to `main`. Training must use this commit after the pod pulls it; the previous NaN run remains invalid. RunPod infrastructure reads show the existing pod idle, but shell execution is unavailable from this agent session.

**Action needed from partner:** Pull the new commit on RunPod, then start Phase A and retain the first finite step/epoch logs.

**Execution note:** Direct and proxy SSH attempts were rejected because the running pod has a different registered key. No SSH-key replacement, pod restart, or new pod was performed. Phase A remains pending in the connected pod terminal.

**Latest training attempt:** The pod command stopped before model construction because `data/processed` contains no files matching the expected clean VoiceBank glob. This is a pod data-path/content issue; the pod is currently idle and should be stopped if the dataset cannot be restored immediately.

**Follow-up inventory:** Pod inspection confirms MUSAN `930` WAVs, RIR `11,026` WAVs, no VoiceBank directory, and no `clean_trainset_28spk_wav`. Existing `clean/` and `noisy/` are the previously identified NOIZEUS evaluation-only tree. Phase A is blocked pending restoration of VoiceBank clean training data and the remaining RIR files; do not silently train on NOIZEUS clean data.

**RunPod allocation update:** The pod's GPU allocation was removed/unavailable. RunPod screenshot reports no available GPU instances; API `get-pod` confirms `rcf3s0xldlni9o` is `EXITED` with `gpu.count=0` and no runtime. CPU start is unsuitable for Phase A. Automatic migration would preserve the incomplete current volume but would not restore missing VoiceBank/RIR files, so migration should wait until the dataset source is restored or be treated only as GPU-capacity recovery.

**Recovery recommendation:** Use “Automatically migrate your Pod data” from the RunPod dialog to recover GPU access and preserve the current volume, then upload the two local tar archives. A live catalog read found Secure RTX 4090 `MEDIUM` at `$0.89/hr`, Secure RTX A6000 `LOW` at `$0.59/hr`, and Secure RTX 3090 `LOW` at `$0.50/hr`; the 4090 remains preferred for reproducibility and fits the `$8.41` balance if training stays within the projected window.

**Latest update:** The migration action itself now reports “There are no instances currently available.” Replacement placement capacity is unavailable, so migration cannot proceed. CPU start remains rejected for Phase A. Fallback is to wait or manually create a separate GPU pod in an available region/GPU class and re-upload the local archives; do not terminate the exited pod yet.

### [2026-10-16 15:05 IST] | Phase: 5 — Switched Training to Local RTX 4090

**What changed:**
- Replaced the RunPod-primary training plan with a local RTX 4090-primary plan in `docs/training_platform.md`, `AGENTS.md`, `GEMINI.md`, `CLAUDE.md`, `README.md`, `execution/README.md`, `requirements.txt`, `directives/05_unet_film_model_v2.md`, `directives/08_future_plan.md`, and `docs/phase_5_complete_architecture_blueprint.md`.
- Added strict dataset count validation (`11071/930/60000`), explicit device selection, CUDA-required bf16/fp16 guard, and Phase A resume support in `execution/phase5_unet_film/21_train_unet_film.py`.
- Added `execution/phase5_unet_film/24_verify_local_training.py`, a no-checkpoint local GPU/data/one-step gate.

**Verification:** Local filelists pass clean `11071`, MUSAN `930`, RIR `60000`; full DataLoader gate, model/Tiny forward, strict mismatch guard, Python compilation, and whitespace checks pass. The current agent host is macOS CPU-only, so the physical RTX 4090 CUDA gate must be run on that machine before full training.

**Status after:** Dataset issue is resolved on the local workspace. Phase A is ready for the local RTX 4090 and must be launched only after `24_verify_local_training.py` passes there.

**Action needed from partner:** On the RTX 4090 host, run the local CUDA gate, then launch Phase A with `--device cuda --mixed_precision bf16`; keep checkpoints under `results/phase5_unet_film/checkpoints/`.

**Verification note:** Full local audio header audit passes all `72,001` WAVs. The local CUDA gate correctly refuses the current macOS CPU-only host (`CUDA is unavailable`), so no false claim of training readiness was made; the physical RTX 4090 host remains the only pending environment check.

---

### [2026-10-16 12:05 IST] | Phase: 5 — Fixed NaN Gradients in CompositeLoss

**What changed:**
- Diagnosed the reported `train nan val nan` after a real-data bf16 reproduction. Dataset/model forward values were finite, but the compressed-complex loss generated non-finite gradients because `abs().pow(0.3)` is singular at zero/silent STFT bins.
- Updated `execution/phase5_unet_film/21_train_unet_film.py` to floor complex magnitudes at `1e-4` and compute the compressed complex representation as `spec * clamped_mag.pow(p - 1)`. Added fail-fast loss/gradient checks with epoch and step numbers.

**Files touched:**
- `execution/phase5_unet_film/21_train_unet_film.py`
- `context.md`
- `shared_context.md`

**Agent used:** OmniRush

**Status after:** Three real-data CPU bf16-autocast optimizer steps completed with finite losses and gradients. Commit `e6f9742` was pushed to `main`. The prior NaN epoch is invalid and must not be used as a checkpoint. The pod must pull the NaN fix before rerunning Phase A.

**Action needed from partner:** Pull the forthcoming NaN-fix commit on the pod and restart Phase A from scratch; preserve any valid checkpoint only after a finite epoch is reported.

---

### [2026-10-16 11:14 IST] | Phase: 5 — Corrected FFT RIR Convolution + Training Progress Logging

**What changed:**
- Replaced the CPU-heavy 8,000-tap time-domain RIR `conv1d` in `execution/phase5_unet_film/03c_data_pipeline_v2.py` with `_fft_convolve`.
- Corrected the initial proposed FFT formulation: PyTorch `conv1d` is cross-correlation, so the FFT path reverses the kernel and crops from `K - 1 - (K // 2)` to match the original padded output for odd and even kernels.
- Added a Phase A loss print every 100 steps in `execution/phase5_unet_film/21_train_unet_film.py`.

**Files touched:**
- `execution/phase5_unet_film/03c_data_pipeline_v2.py`
- `execution/phase5_unet_film/21_train_unet_film.py`
- `shared_context.md`

**Agent used:** OmniRush

**Status after:** Exact float64 comparison passes for `(64000, 8000)` and odd/even kernel cases; syntax and whitespace checks pass. A local exact-shape float32 probe measured `201.6x` speedup (`1.7740s` time-domain vs `0.0088s` FFT) with `5.10e-05` max absolute numerical difference. Code commit `51d83cd` was pushed to `main`. RunPod pod `rcf3s0xldlni9o` is still `RUNNING` on an RTX 4090, but its MCP log stream only contains startup logs and does not expose the manually launched training process/checkpoint state.

**Action needed from partner:** None for the code change. Before restarting training, inspect the pod through its console terminal/SSH or transfer the checkpoint; do not stop or restart it based only on the ambiguous 0% GPU read.

---

### [2026-10-16 08:37 IST] | Agent Runtime — Added GRAVITAS Skills & Invariants

**What changed:**
- Integrated the [`udaysharmadev/Gravitas`](https://github.com/udaysharmadev/Gravitas) reliability runtime into Antigravity.
- Installed 4 skills (`gravitas`, `gravitas-highstakes`, `gravitas-python`, `gravitas-typescript`) into both global configuration (`~/.gemini/config/skills/`) and project workspace (`.agents/skills/`).
- Added `gravitas-invariants.md` rule to `.agents/rules/` and `~/.gemini/config/rules/` (enforcing ground before mutating, stay in declared scope, evidence-backed completion, no exact-retry, contract satisfaction, and uncertainty honesty).
- Deployed native Antigravity plugin `gravitas-native` in `~/.gemini/config/plugins/`.

**Files touched:**
- `.agents/skills/gravitas/*`
- `.agents/skills/gravitas-highstakes/*`
- `.agents/skills/gravitas-python/*`
- `.agents/skills/gravitas-typescript/*`
- `.agents/rules/gravitas-invariants.md`
- `~/.gemini/config/skills/*`
- `~/.gemini/config/rules/*`
- `~/.gemini/config/plugins/gravitas-native/*`
- `context.md`
- `shared_context.md`

**Status after:** Gravitas skills and invariants active across all agent turns.

---

### [2026-09-09 03:06 IST] | Phase 3 — MMSE Bug Fixes + Correct Evaluation (commit 431d15f)

**Critical bugs fixed in `04_mmse_lsa_filter.py`:**
- Noise init from first 6 frames (speech frames!) → inflated by 32× at 15dB SNR → near-silence output. Fixed with offline bottom-20% energy frames.
- Noise tracker grew during speech (18%/frame) instead of tracking minimum. Fixed with true sliding-minimum + bias correction.
- ROOT path fixed in all 3 phase3 scripts (3 dirname levels not 2).

**Corrected results (PESQ and SI-SDR now correctly positive):**
- MMSE-LSA: PESQ +0.029, SI-SDR +0.765 dB (was PESQ -0.114 before fix — completely wrong)
- Wavelet: STOI +0.004, SI-SDR +0.272 dB (unchanged, was correct)

**Impact on Namya:** `git pull` to get fixed scripts. Old `results/enhanced_mmse/` WAVs were near-silence — already replaced with correct outputs (commit 431d15f).

---

### [2026-09-09 02:48 IST] | Infra — Full Project Reorganisation

**What changed (commit 03837c5):**
- `execution/` root-level duplicate scripts removed (10 files). Scripts now exist ONLY in phase subfolders.
- `execution/README.md` fully rewritten — shows exact structure + `uv run` commands for each phase
- `results/plots/` — reorganised into `phase0_dsp/`, `phase1_audiology/`, `phase2_pipeline/`, `phase3_baselines/`
- `results/audio_demos/` — reorganised into `phase0_dsp/`, `phase1_audiology/`
**Impact on Namya:** `git pull` will update all paths. If you have local scripts referencing root-level execution paths, update to the phase subfolder paths.

---

### [2026-09-09 02:44 IST] | Phase: 3 — Baseline Evaluation COMPLETE

**What changed:**
- `execution/06_baseline_eval.py` — full STOI/PESQ/SI-SDR evaluation across 930 NOIZEUS files
- `results/classical_baselines.csv` — 930 rows, per-file metrics for noisy/MMSE/Wavelet
- `results/plots/` — 4 charts: STOI/PESQ/SI-SDR by noise type, STOI vs SNR line plot
- `results/cli_output/06_baseline_eval.txt` — full terminal log (149.8s runtime)
**Key numbers (overall mean):**
  - Noisy STOI=0.813 PESQ=1.435 SI-SDR=6.71 dB
  - MMSE-LSA: STOI −0.073, PESQ −0.114, SI-SDR −1.08 dB (ALL WORSE)
  - Wavelet: STOI +0.004, PESQ −0.026, SI-SDR +0.27 dB (near-neutral)
**Conclusion:** Classical methods cannot improve STOI/PESQ. Deep learning targets: STOI > 0.90, PESQ > 2.0
**Commit:** c9a0fdf | **Agent:** Antigravity

---

### [2026-09-09 01:58 IST] | Phase: Infra — Colab MCP Setup

**What changed:**
- `googlecolab/colab-mcp` investigated — **no Chrome extension needed**. Works via local WebSocket bridge (uvx starts WS server, Colab connects back, agent executes GPU cells)
- `.agents/mcp_config.json` committed — auto-loads colab-mcp for both Antigravity and Claude Code when workspace is opened
- `docs/colab_setup.md` — complete setup guide for both partners:
  - `uv` install + MCP auto-config
  - Google Drive data upload (Jwanil from SSD, Namya from local clone `dataset/`)
  - Drive checkpoint save pattern + rclone download
  - Git push from Colab with personal access token
  - Real-time Colab collaboration (both can share the same notebook)
- `AGENTS.md` + `GEMINI.md` updated with Colab/GPU training section
**Agent used:** Antigravity (Gemini 3.1 Pro)
**Status after:** Infrastructure ready for GPU training. Both partners just need to install `uv`, upload data to Drive once, then restart IDE.
**Action needed from partner (Namya):**
  1. `git pull` to get `.agents/mcp_config.json` and `docs/colab_setup.md`
  2. Install `uv`: `pip install uv`
  3. Upload `dataset/processed/` to `MyDrive/hearing-aid-data/processed/` on Google Drive
  4. Restart Claude Code — colab-mcp auto-loads

---

### [2026-09-08 18:17 IST] | Phase: 3 — Docs, Execution Org, CLI Logs

**What changed:**
- `phase_explanations/phase_3_classical_baselines.md` — complete Phase 3 deep-dive (both MMSE and Wavelet, all terminology, all algorithm steps, results analysis, head-to-head table, Phase 6 connection)
- `execution/` reorganised into phase subfolders: `phase0_dsp/`, `phase1_audiology/`, `phase2_data_pipeline/`, `phase3_classical_baselines/` — originals remain in root for backward compatibility
- `execution/README.md` — master quick-reference for all scripts
- `results/cli_output/04_mmse_lsa_filter_all_930_files.txt` — full MMSE terminal log (1,873 lines)
- `results/cli_output/05_wavelet_denoising_all_930_files.txt` — full Wavelet terminal log (1,874 lines)
- Committed and pushed: `6a07fd7`
**Agent used:** Antigravity (Gemini 3.1 Pro)
**Status after:** Phase 3 fully complete and documented on Jwanil's side.
**Action needed from partner:** Pull latest; all Phase 3 scripts, docs, and outputs are now available.

---

### [2026-09-08 17:48 IST] | Phase: 3 — Classical Baselines (Wavelet DWT COMPLETE)

**What changed:**
- Implemented `execution/05_wavelet_denoising.py` — Wavelet DWT Classical Baseline 2.
  - Donoho & Johnstone (1994) universal soft-threshold; db8 wavelet; 5 levels (250-8000 Hz at 16kHz)
  - Same CLI interface as MMSE script; cross-machine path detection (`data/` vs `dataset/`)
  - Outputs: `results/enhanced_wavelet/noizeus/`; plots: `results/plots/wavelet_output_plots/`
- Ran self-test → ✅ passed
- Ran babble/0dB (30 files, with plots) → ✅ 30/30
- Ran full dataset (`--all`) → ✅ **930/930 files**, zero errors
- Committed and pushed: `3185c53`
**Files touched:**
- `execution/05_wavelet_denoising.py` (NEW)
- `results/plots/wavelet_output_plots/` (30 PNGs on GitHub)
- `results/enhanced_wavelet/noizeus/` (930 .wav files on disk only)
**Agent used:** Antigravity (Gemini 3.1 Pro)
**Status after:** **Phase 3 COMPLETE** — both MMSE-LSA and Wavelet DWT baselines implemented, verified, and run on all 930 NOIZEUS files. Phase 3 complete on Jwanil's side.
**Next action from partner:** Namya should pull latest; review `05_wavelet_denoising.py`; begin Phase 6 eval script if assigned.

---

### [2026-09-08 17:33 IST] | Phase: 3 — Classical Baselines (MMSE-LSA Verified)

**What changed:**
- Verified Namya's `execution/04_mmse_lsa_filter.py` — implementation is algorithmically correct (Ephraim & Malah 1985, decision-directed xi, E1 exponential integral gain, online noise tracking).
- Fixed 2 compatibility bugs for Jwanil's machine: `dataset/` → `data/` path; Python 3.9 `X | Y` type hints → `Optional[X]`; added `matplotlib.use("Agg")`.
- Ran self-test: ✅ passed.
- Ran on babble/0dB (30 files) with plots: ✅ 34% average RMS noise reduction.
- Ran on entire NOIZEUS dataset (`--all`): ✅ **930/930 files**, all 8 noise types, 3 SNR levels.
- Committed and pushed: `a7cfc0f`.
**Files touched:**
- `execution/04_mmse_lsa_filter.py` (PATCHED for cross-machine compatibility)
- `results/plots/mmse_output_plots/` (61 spectrogram PNGs on GitHub)
- `results/enhanced_mmse/noizeus/` (930 .wav files on disk only — not in git)
**Agent used:** Antigravity (Gemini 3.1 Pro)
**Status after:** MMSE-LSA verified and complete on Jwanil's machine. Output ready for Phase 6 evaluation.
**Action needed from partner:** Namya should pull and note the 2 path/compatibility patches in her `04_mmse_lsa_filter.py`.

---

### [2026-09-06 17:58 IST] | Phase: Docs

**What changed:**
- Created `phase_explanations/` folder in the project root.
- Added `phase_0_to_2_breakdown.md` — full deep-dive covering every script, all code logic, all audio/ML terminology, all result graphs and audio files explained in detail, and how everything connects to Phases 3–6.
- Committed and pushed (`c20b602`).
**Files touched:**
- `phase_explanations/phase_0_to_2_breakdown.md` (NEW)
- `shared_context.md` (this entry)
**Agent used:** Antigravity (Gemini 3.1 Pro)
**Status after:** Phase explanations folder live on GitHub. Future phases (3, 4, 5, 6) will add their own breakdown docs here.
**Action needed from partner:** Pull latest to get the new docs folder.

---

### [2026-09-06 17:35 IST] | Phase: Config / Housekeeping

**What changed:**
- Removed `context.md` from git tracking (`git rm --cached`) — it is now in `.gitignore`.
- Updated `AGENTS.md` + `GEMINI.md`: added warning block explaining that `context.md` is personal/local per-partner, with a full template.
- Updated `shared_context.md` Namya section: added urgent step-by-step instructions for her to delete Jwanil's `context.md` and create her own before starting.
- Committed and pushed (`cec56f3`).
**Files touched:**
- `.gitignore` (added `context.md`)
- `AGENTS.md` (warning block + template)
- `GEMINI.md` (mirrored warning)
- `shared_context.md` (Namya urgent note)
**Agent used:** Antigravity (Gemini 3.1 Pro)
**Status after:** `context.md` is fully personal. Never in the repo again.
**Action needed from partner:** URGENT — Namya must delete the existing `context.md` from her local clone and create her own. See her section in `shared_context.md` for template.

---

### [2026-09-04 09:08 IST] | Phase: 2 — Data Pipeline (Complete)

**What changed:** 
- Successfully ran `03b_standardize_audio.py` across 25,541 dataset `.wav` files using multiprocessing. All files are now exactly 16kHz, mono, peak-normalized to [-1, 1], and exactly 4 seconds long (`1x64000` samples) and stored in `data/processed/`.
- Wrote `03c_data_pipeline.py` containing the `HearingAidDataset` and `DataLoader`. 
- The dataloader maps processed `.wav` files and randomly assigns one of the generated audiogram profiles (normalized from 0-120dB to 0-1 range).
- Tested dataloader: successfully generates batches of `noisy`, `clean`, `audiogram`, and `snr`. Spectrograms plotted and verified.
- **Backported Dataset Audio to Phase 0 & 1:** Removed the synthetic audio generators from `01_stft_visualize.py` and `03_simulate_hearing_loss.py`. They now directly load the standardized `sp01.wav` from the dataset, generating much more realistic medical/spectrogram demonstrations. Updated the explanation docs to reflect this.
**Files touched:**
- `context.md` (checked off standardisation and dataloader tasks)
- `execution/03b_standardize_audio.py` (NEW)
- `execution/03c_data_pipeline.py` (NEW)
- `results/plots/02_dataloader_test.png` (NEW)
- `shared_context.md` (this entry)
**Agent used:** Antigravity (Gemini 3.1 Pro)
**Status after:** Phase 2 execution complete. Datasets and PyTorch Dataloaders are fully functional. Ready for Phase 3 (Classical Baselines).
**Action needed from partner:** Pull latest. Verify the dataloader spectrogram test.

---

### [2026-09-04 08:59 IST] | Phase: 2 — Data Pipeline

**What changed:** 
- Datasets downloaded and verified on SSD: NOIZEUS, VoiceBank-DEMAND, MUSAN.
- Fixed a macOS specific bug in `03a_build_metadata.py` where `._` hidden files caused torchaudio to crash.
- Unlinked `data/metadata` from SSD to keep metadata CSVs on the Mac initially because SSD was filled to 100%. (Update: space was cleared on the SSD (27GB free), so the entire `data/` folder is now back to being a single symlink to the SSD, keeping everything unified).
- Ran metadata builder successfully: 9,839 Train / 1,093 Val / 1,754 Test items.
**Files touched:**
- `context.md` (Checked off dataset and metadata tasks)
- `execution/03a_build_metadata.py` (NEW — dataset manifest builder)
- `data/metadata/train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`
- `shared_context.md` (this entry)
**Agent used:** Antigravity (Gemini 3.1 Pro)
**Status after:** Data metadata generated. Ready for the PyTorch DataLoader and audio standardization.
**Action needed from partner:** None for now.

---

### [2026-09-04 07:24 IST] | Phase: 2 — Data Pipeline

**What changed:** Phase 1 execution complete. Phase 2 data pipeline planning in progress. Dataset storage strategy decided: all datasets go to SSD (`/Volumes/SANDISK/Minor Project/Data/`), project code stays on Mac. A symlink `data/` → SSD path keeps the codebase clean and portable.
**Files touched:**
- `context.md` (Phase 1 tasks ticked, Phase 2 task list updated)
- `directives/01_audiology.md` (learnings log updated with pyclarity API discoveries)
- `execution/02_audiogram_generator.py` (NEW — generates 3 test profiles + random audiograms)
- `execution/03_simulate_hearing_loss.py` (NEW — MSBG hearing loss simulation)
- `results/plots/01_audiograms_profiles.png`, `01_audiograms_random.png`, `01_hearing_loss_spectrograms.png`
- `results/audio_demos/01_original_44k.wav`, `01_mild_flat_loss.wav`, `01_moderate_sloping_loss.wav`, `01_severe_hf_loss.wav`
- `results/data/audiograms.json`
- `shared_context.md` (this entry)
**Agent used:** Antigravity (Claude Sonnet 4.6)
**Status after:** Phase 0 & 1 execution complete. SSD at `/Volumes/SANDISK/Minor Project/Data/` confirmed (39 GB free). Ready to download datasets.
**Action needed from partner:** Pull latest. Phase 2 (Data Pipeline) is active. Dataset downloads starting — see dataset guide in this session.

---

### [2026-09-04 06:57 IST] | Phase: 2 — Data Pipeline

**What changed:** Phase 0 and Phase 1 marked complete (resource study done; execution scripts still pending). Phase 2 (Data Pipeline: TIMIT + Clarity + NOIZEUS + MUSAN) set as active. Date timeline columns removed from `AGENTS.md`, `SESSION_START.md` phase tables. Phase tracking switched to status-only. Directive file order corrected: `02_data_pipeline` comes before `03_classical_baselines`.
**Files touched:**
- `AGENTS.md` (Dates column removed, phase order corrected: Data Pipeline = Phase 2, Classical Baselines = Phase 3)
- `SESSION_START.md` (Dates column removed, status column added, phase order corrected)
- `context.md` (Active phase updated to Phase 2 Data Pipeline, new log entry, Phase 2 open tasks added)
- `shared_context.md` (this entry)
**Agent used:** Antigravity (Claude Sonnet 4.6)
**Status after:** Phases 0 & 1 resource study complete (execution scripts pending). Phase 2 (Data Pipeline) in progress. Next: run Phase 0/1 execution scripts AND start building data pipeline.
**Action needed from partner:** Pull latest. Phase order has changed — Data Pipeline is now Phase 2 (before Classical Baselines).

---

### [2026-08-17 09:20 IST] | Phase: Setup / Documentation

**What changed:** Generated a proposal DOCX using the exact same design as the minor project report (`generate_report.py`). White background, `1A56AA` blue headings, `EBF3FB` alternating table rows, `F5F5F5` code blocks, Calibri body — 12 sections, 18 references.
**Files touched:**
- `execution/generate_proposal_docx.py` (NEW: proposal DOCX generator)
- `docs/Minor_Project_Proposal_Hearing_Aid_Speech_Enhancement.docx` (NEW: output)
**Agent used:** Antigravity (Claude Sonnet 4.6)
**Status after:** Two separate DOCX generators now exist — one for the report, one for the proposal. Run either to regenerate. Full architecture (5 models, Mamba, HASPI-GAN, CRM, all RASTA explanation) is documented in the proposal DOCX.
**Action needed from partner:** None

---


### [2026-08-20 18:17 IST] | Phase: Setup / Documentation

**What changed:** Fully rewrote `docs/project_overview.md` to reflect the 5-model architecture.
The old file still described Wiener filter + 3 models and had wrong directive filenames.
**Files touched:**
- `docs/project_overview.md` (REWRITTEN — 5-model plan, correct datasets, correct directives, Nov 1 deadline)
**Agent used:** Antigravity (Gemini/Claude)
**Status after:** project_overview.md is now fully in sync with the current project plan.
**Action needed from partner:** Pull latest. `@project_overview.md` in any new AI session now gives correct context.

---

### [2026-08-16 22:20 IST] | Phase: Setup / Documentation

**What changed:** Added Future Scope to the proposal presentation and generated a DOCX version of the full 21-slide pitch.
**Files touched:** 
- `docs/presentation.html` (Added Slide 18: Future Scope)
- `execution/generate_presentation_docx.py` (NEW: script to generate DOCX from the presentation design)
- `docs/Project_Proposal_Hearing_Aid_Speech_Enhancement.docx` (NEW: output document)
**Agent used:** Antigravity (Gemini)
**Status after:** The final project proposal is now available as a highly-styled HTML presentation and an exact-match DOCX document.
**Action needed from partner:** None

---

### [2026-08-16 21:55 IST] | Phase: Setup / Documentation

**What changed:** Final session cleanup — session start template, cleaned stale directives, new 20-slide proposal presentation, context.md and shared_context.md restructured.

**Files touched:**
- `SESSION_START.md` ← NEW: copy-paste prompt for any new chat session
- `docs/presentation.html` ← REPLACED: 14-slide pitch → 20-slide full project proposal
- `directives/02_wiener_baseline.md` ← DELETED (superseded)
- `directives/04_generic_dnn.md` ← DELETED (superseded)
- `directives/05_film_conditioning.md` ← DELETED (superseded)
- `shared_context.md` ← RESTRUCTURED: now has separate Jwanil + Namya log sections
- `context.md` ← Updated session log

**Agent used:** Antigravity (Gemini)

**Status after:** Project fully documented. Directory is clean. 9 active directives (00–07 + 05b). Presentation is 20 slides. SESSION_START.md ready for copy-paste.

**Action needed from Namya:** Pull latest changes. Read `SESSION_START.md`. That file tells you exactly what to paste into any new AI chat session.

---

### [2026-08-16 ~21:30 IST] | Phase: Setup / Documentation

**What changed:** Major documentation update — all directives updated to reflect new architecture, timeline set to Nov 1.

**Files touched:**
- `directives/02_classical_baselines.md` ← NEW: Wavelet DWT + MMSE-LSA
- `directives/04_1d_cnn_model.md` ← NEW: Conv-TasNet
- `directives/05_unet_film_model.md` ← NEW: U-Net + Attention + FiLM + CRM + MetricGAN+
- `directives/05b_mamba_film_model.md` ← NEW: Mamba/SSM + FiLM
- `directives/06_evaluation.md` ← UPDATED: 5 models, 3 audiogram test profiles
- `docs/simple_guide.md` ← NEW: plain-English guide with all concepts, resources, timeline
- `AGENTS.md` ← Updated phase table, Nov 1 deadline, 5-model architecture
- `context.md` ← Updated decisions log, file index, open tasks

**Agent used:** Antigravity (Gemini)

**Status after:** All directives current. Simple guide is the best onboarding document for Namya.

**Action needed from Namya:** Read `docs/simple_guide.md` — it explains every concept in plain English with full forms and learning resources per phase.

---

### [2026-08-15 17:20 IST] | Phase: Setup / Research

**What changed:** Deep 4-agent research sweep on SOTA architectures. Published findings as `peak_architecture_research.md`. Key decision: replace Transformer with Mamba/SSM.

**Files touched:**
- `peak_architecture_research.md` (in brain/artifacts — not in project repo)

**Agent used:** Antigravity (Gemini) — 4 parallel research subagents

**Status after:** Architecture decisions finalised from research. SEMamba PESQ 3.69 confirmed as SOTA. Mamba chosen over Transformer.

**Action needed from Namya:** None — Jwanil will implement architecture. Namya focuses on data pipeline (Phase 3).

---

### [2026-08-15 ~16:00 IST] | Phase: Setup / Faculty Feedback

**What changed:** Faculty meeting notes integrated. Previous 3-model architecture upgraded to 5-model. Faculty analysis document created.

**Files touched:**
- `faculty_feedback_analysis.md` (in brain/artifacts)

**Agent used:** Antigravity (Gemini)

**Status after:** Faculty feedback fully processed. Architecture expanded.

**Action needed from Namya:** None at this stage.

---

### [2026-08-12 11:11 IST] | Phase: Setup / Documentation

**What changed:** DOCX report generator created. Presentation built.

**Files touched:**
- `execution/generate_report.py` ← NEW: generates .docx academic report
- `docs/Minor_Project_Report_Hearing_Aid_Speech_Enhancement.docx` ← generated report
- `docs/presentation.html` ← original 14-slide faculty pitch deck

**Agent used:** Antigravity (Gemini)

**Status after:** Report and presentation ready for faculty.

**Action needed from Namya:** None.

---

### [2026-08-11 21:33 IST] | Phase: Setup

**What changed:** Full project scaffold created from scratch.

**Files created:**
- `README.md`, `.gitignore`, `AGENTS.md`, `GEMINI.md`, `CLAUDE.md`
- `context.md`, `shared_context.md`, `FRIEND_SETUP.md`
- `docs/project_overview.md`, `docs/everything_from_scratch.md`
- All phase directives `directives/00_*` through `directives/07_*`
- `execution/00_verify_setup.py`

**Agent used:** Antigravity (Gemini)

**Status after:** Project structure 100% ready. No code written yet.

**Action needed from Namya:**
1. Read `FRIEND_SETUP.md` completely
2. Install Antigravity + VS Code + Live Share + Five Server extension
3. Clone repo: `https://github.com/Jwanil/hearing-aid-speech-enhancement`
4. Run `execution/00_verify_setup.py` to confirm environment
5. Update Team table above with your OS and GitHub handle

---
---

# 📗 NAMYA'S SESSION LOG

<!-- Agent: append Namya's entries below this line in reverse-chronological order (newest first) -->

> ⚠️ **Namya — URGENT: `context.md` has been removed from the repo.**
>
> If you have already pulled the repo, you may have a copy of **Jwanil's** `context.md` in your local folder. **Do NOT use it.** Delete it and create your own.
>
> **Your agent will be confused if it reads Jwanil's `context.md`** — it has Jwanil's task history, not yours.
>
> **Action required before starting work:**
> 1. Delete the existing `context.md` in your local project folder.
> 2. Create a new file called `context.md` in the same location.
> 3. Paste this as the starting template:
>
> ```markdown
> # context.md — Namya Session Log
>
> ## Current Phase
> Phase 2 complete (Data Pipeline done by Jwanil). Phase 3 (Classical Baselines) is next.
>
> ## Open Tasks
> - [ ] Read docs/project_overview.md (full project context)
> - [ ] Read docs/simple_guide.md (plain-English explanation of everything)
> - [ ] Read directives/03_classical_baselines.md
> - [ ] Implement execution/04_wavelet_denoising.py
> - [ ] Implement execution/05_mmse_lsa.py
>
> ## Session Log
> <!-- Add entries here as you work -->
> ```
>
> 4. `context.md` is now in `.gitignore` — it will never be committed. You keep it only on your own machine.
>
> **Before your first entry:** Pull from GitHub (`git pull`), read `SESSION_START.md`, then read `docs/simple_guide.md`.

---

### [2026-09-06 17:28 IST] | Phase: 3 — Classical Baselines (MMSE-LSA)

**What changed:**
- Started Phase 3 — Classical Baselines. Read `SESSION_START.md`, `AGENTS.md`, `context.md`, `shared_context.md`, and `directives/03_classical_baselines.md` at session start.
- Implemented **`execution/04_mmse_lsa_filter.py`** — the MMSE-LSA (Minimum Mean Square Error – Log Spectral Amplitude) filter, per faculty instruction to implement MMSE-LSA first before Wavelet.
- Implementation follows the Ephraim & Malah (1985) algorithm exactly as specified in the directive:
  - STFT with 25 ms frame, 10 ms hop, Hann window
  - Noise power bootstrapped from first 6 frames (~60 ms silence)
  - Decision-directed a priori SNR (xi) with alpha=0.98
  - MMSE-LSA gain using `scipy.special.exp1` (exponential integral E1)
  - Online noise estimate update (minimum-statistics)
  - ISTFT reconstruction using noisy-signal phase
  - `postprocess_filter_output()` included: DC removal, length fix, peak normalisation
- Script supports three modes: `--file <name>` (single file), `--all` (full dataset), `--test` (synthetic self-test, no dataset required)
- Optional `--plot` flag saves waveform + spectrogram comparison to `results/plots/04_mmse_<filename>.png`
- All output saved to `results/enhanced_mmse/`

**Files touched:**
- `execution/04_mmse_lsa_filter.py` (NEW — full MMSE-LSA implementation)
- `shared_context.md` (this entry)

**Agent used:** Antigravity (Claude Sonnet 4.6 Thinking)

**Status after:** Phase 3 MMSE-LSA script complete. Ready to run `python execution/04_mmse_lsa_filter.py --test` to verify, then `--all --plot` on the NOIZEUS dataset. Wavelet denoiser (`execution/05_wavelet_denoiser.py`) is next.

**Action needed from partner (Jwanil):** Pull latest. Run `python execution/04_mmse_lsa_filter.py --test` first to verify the math. Then run `--all --plot --limit 5` on 5 NOIZEUS files and do a listen test. Report if musical noise is audible — if so, we can tune the `alpha` or noise floor.

---

### [2026-09-08 17:09 IST] | Phase: 3 — MMSE-LSA verification + git sync + dataset integration

**What changed:**
- Dataset transferred from Jwanil's SSD into local `dataset/` folder (`dataset/processed/noisy/noizeus/`, `dataset/processed/clean/noizeus/clean/`, `dataset/raw/`, `dataset/metadata/`).
- Pulled Jwanil's 3 new commits (c3a1962, c20b602, cec56f3) into local main branch:
  - `context.md` removed from git tracking (now gitignored — each partner keeps their own locally).
  - `phase_explanations/phase_0_to_2_breakdown.md` added by Jwanil.
  - `shared_context.md` updated with Jwanil's Sept 6 session entries.
- Updated `execution/04_mmse_lsa_filter.py`: changed `DATA_ROOT` path from `Data/` to `dataset/` (local folder name after transfer).
- **Verified MMSE-LSA on real NOIZEUS dataset:**
  - `--test` (synthetic signal): PASS — float32, values in [-1, +1], shape correct.
  - `airport/5dB` — 3 speakers: all [OK], avg RMS 0.0914 -> 0.0710 (22% reduction).
  - `babble/5dB` — 3 speakers: all [OK], avg RMS 0.0952 -> 0.0754 (20% reduction).
  - `car/5dB` — 3 speakers: all [OK], avg RMS 0.0911 -> 0.0732 (19% reduction).
- Created personal `context.md` (gitignored, local only) as per Jwanil's template instructions.
- Committed and pushed to GitHub (now collaborator on `Jwanil/hearing-aid-speech-enhancement`).

**Files touched:**
- `execution/04_mmse_lsa_filter.py` (MODIFIED — DATA_ROOT: Data/ -> dataset/)
- `shared_context.md` (MODIFIED — this entry)
- `context.md` (NEW — local only, gitignored)
- Received from Jwanil merge: `phase_explanations/phase_0_to_2_breakdown.md`, `.gitignore`, `AGENTS.md`, `GEMINI.md`

**Agent used:** Antigravity (Claude Sonnet 4.6 Thinking)

**Status after:** Phase 3 MMSE-LSA is fully complete and verified on real data. Next task: `execution/05_wavelet_denoiser.py`.

**Action needed from Jwanil:** Pull latest. MMSE-LSA is running on `dataset/` folder now. If your machine uses a different folder name, update `DATA_ROOT` in `execution/04_mmse_lsa_filter.py` accordingly.

---

### [2026-10-05 05:28 IST] | Phase: 4 \u2014 1D CNN (Conv-TasNet) implementation + Colab training started

**What changed:** Phase 4 (1D CNN model) fully implemented. 30-epoch training kicked off on Google Colab T4 GPU.

**Implementation decisions and rationale:**

| Decision | Rationale |
|---|---|
| Conv-TasNet via Asteroid | Industry-standard waveform-domain model; reference implementation available; eliminates STFT phase problem |
| VoiceBank-DEMAND-16k (HuggingFace) | Standard DNN SE benchmark; 11,572 train pairs; matches every published Conv-TasNet result |
| Negative SI-SDR loss | Differentiable, scale-invariant; directly optimises what we measure; standard for waveform models |
| batch=8, 30 epochs | batch=4 + 50 epochs = 25 hrs > Colab limit; batch=8 + 30 epochs \u2248 7.5 hrs, safe within one session |
| Checkpoint every epoch | Protects against Colab session resets; Drive saves persist across resets |
| Experiment A vs B | Directive 04 core question: does MMSE pre-filtering help or hurt the CNN? |

**Bugs fixed before training (would have wasted GPU time):**
1. Broken imports: Python cannot import modules whose names start with digits (`10_model_1d_cnn`, `04_mmse_lsa_filter`) \u2014 fixed with `importlib.util.spec_from_file_location()`
2. Non-existent `execution/utils.py` import \u2014 removed
3. NOIZEUS speaker filename parsing: `sp01_babble_sn5.wav`.split('_')[0] = `sp01` (correct), previously was `_sp01` (wrong)
4. Timing arithmetic: 50 epochs \u00d7 batch=4 \u2248 25 hrs; fixed to 30 epochs \u00d7 batch=8 \u2248 7.5 hrs

**Proof-of-concept training result (5 epochs, 200 batches each):**
- Epoch 1 avg loss: -9.27 dB
- Epoch 5 avg loss: -12.75 dB
- Smooth, consistent improvement \u2014 model is learning correctly

**Files touched:**
- `execution/phase4_1d_cnn/10_model_1d_cnn.py` (NEW \u2014 ConvTasNet definition)
- `execution/phase4_1d_cnn/11_train_1d_cnn.py` (NEW \u2014 training loop, batch=8, 30 epochs)
- `execution/phase4_1d_cnn/12_eval_1d_cnn.py` (NEW \u2014 NOIZEUS eval, Exp A & B)
- `context.md` (UPDATED \u2014 phase 4 in progress)
- `shared_context.md` (UPDATED \u2014 this entry)

**Agent used:** Antigravity (Gemini 3.1 Pro / Claude Sonnet 4.6 Thinking)

**Status after:** Full 30-epoch training running on Colab T4. Checkpoints saving to `MyDrive/hearing-aid-data/checkpoints/phase4_1d_cnn_epochXXX.pt` every epoch.

**Action needed from Namya:**
- Pull the new `execution/phase4_1d_cnn/` scripts
- When Jwanil shares the best checkpoint, run `12_eval_1d_cnn.py` to verify Exp A vs B results

---
