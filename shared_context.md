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

<!-- Agent: append Jwanil's entries below this line in reverse-chronological order (newest first) -->

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
