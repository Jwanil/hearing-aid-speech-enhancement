# context.md — Per-Session Task Log

> **AGENT: Read this file at the start of every session. Update it after every significant action.**
> This is the single source of truth for what has been done, what is in progress, and what is next.

---

## ⚠️ Context Window Health — Read After Every Prompt

After completing each prompt, run this internal check:

```
CONTEXT HEALTH CHECK (run silently after every response):
[ ] Did I read this file at session start?
[ ] Is my understanding of the current phase correct?
[ ] Are the files I'm referencing real (not hallucinated)?
[ ] Is my reasoning consistent with the last 3 responses?
[ ] Have I updated this log after my last action?

WARNING THRESHOLD: ~80 messages in session, or any detected inconsistency.
ACTION: Output the context-reset block from AGENTS.md and stop.
```

---

## Current Project Phase

**Active Phase:** Phase 0 — Setup & DSP Fundamentals  
**Current Task:** Project scaffolding  
**Next Task:** Install dependencies, run STFT visualization script

---

## Project Status Overview

| Phase | Name | Status | Notes |
|-------|------|--------|-------|
| 0 | Audio/DSP Fundamentals | 🟡 In Progress | Project setup done |
| 1 | Audiology + Audiograms | ⬜ Not Started | |
| 2 | Classical Baselines: Wavelet DWT + MMSE-LSA | ⬜ Not Started | Replaces Wiener filter (faculty feedback) |
| 3 | Data Pipeline (TIMIT + Clarity + NOIZEUS + MUSAN) | ⬜ Not Started | |
| 4 | 1D CNN Model (Conv-TasNet) | ⬜ Not Started | Waveform-domain DL baseline |
| 5 | U-Net + Attention + FiLM | ⬜ Not Started | Core contribution — complex masking + MetricGAN+ |
| 5b | Mamba/SSM + FiLM | ⬜ Not Started | SOTA model — replaces Transformer plan |
| 6 | Full Evaluation | ⬜ Not Started | All 5 models, 3 audiogram profiles |
| 7 | Report + Demo | ⬜ Not Started | |

**Legend:** ✅ Done | 🟡 In Progress | ⬜ Not Started | ❌ Blocked

---

## Environment

- **OS:** macOS
- **Python:** (set when installed)
- **CUDA available:** (check when GPU setup)
- **Workspace path:** `/Users/jwanilmodi/Web development/hearing-aid-speech-enhancement`
- **Key dependencies:** torch, torchaudio, asteroid, speechbrain, pyclarity, pystoi, pesq, pywavelets, mamba-ssm
- **Deadline:** November 1, 2026

---

## Session Log

<!-- Agents: append a new entry below after every significant action. Format:
### [YYYY-MM-DD HH:MM] — [Author: You/Partner] — [Action summary]
Details of what was done, why, and any findings.
-->

---

### [2026-08-11 21:33] — Setup — Project scaffold created

**Author:** Jwanil (via Antigravity)

**Actions:**
- Created project directory at `Web development/hearing-aid-speech-enhancement/`
- Initialized git repository
- Created `README.md`, `.gitignore`
- Created directory structure: `docs/`, `directives/`, `execution/`, `src/`, `notebooks/`, `results/`, `.tmp/`
- Copied `presentation.html` and `everything_from_scratch.md` to `docs/`
- Created `AGENTS.md`, `CLAUDE.md`, `GEMINI.md` (mirrored agent instructions)
- Created `context.md` (this file), `shared_context.md`, `docs/project_overview.md`
- Created all phase directives in `directives/`
- Created `FRIEND_SETUP.md`

**Status:** Project structure complete. Ready to begin Phase 0.

**Next action:** Install Python dependencies, run `execution/00_verify_setup.py` to confirm environment.

---

### [2026-08-12 11:11] — Jwanil — DOCX report generator + original presentation

**Actions:**
- Created `execution/generate_report.py` — programmatic .docx academic report generator
- Generated `docs/Minor_Project_Report_Hearing_Aid_Speech_Enhancement.docx`
- Built 14-slide faculty pitch deck at `docs/presentation.html`

**Status:** Documentation complete for initial faculty pitch.

---

### [2026-08-15 16:00] — Jwanil — Faculty meeting feedback integrated

**Actions:**
- Received faculty notes: replace Wiener filter, add MMSE/Wavelet, add 1D CNN, add TIMIT/NOIZEUS datasets
- Created `faculty_feedback_analysis.md` (artifact — not in repo)
- Expanded architecture from 3-model to 5-model comparison

**Status:** Architecture decisions updated. Deep research sweep initiated.

---

### [2026-08-15 17:00–17:30] — Jwanil — Deep SOTA research sweep (4 parallel agents)

**Actions:**
- Launched 4 parallel research subagents covering: SOTA architectures, hearing-aid ML, advanced techniques, datasets/evaluation
- Key finding: SEMamba (Mamba/SSM) achieves PESQ 3.69 — beats all Transformers AND is linear O(T) complexity
- Decision: replace Transformer plan with Mamba/SSM
- Created `peak_architecture_research.md` artifact with full synthesis
- Added 6 research-backed upgrades: CRM (Complex Ratio Mask), MetricGAN+ with HASPI discriminator, ERB subband FiLM, knowledge distillation, CoNNear, Mamba

**Status:** Final architecture decided. Research complete.

---

### [2026-08-16 21:15–21:55] — Jwanil — Full directive update, simple guide, session template, new presentation

**Actions:**
- Created `docs/simple_guide.md` — plain-English guide: all concepts with full forms, analogies, resources per phase, timeline, what success looks like
- Created `directives/02_classical_baselines.md` — Wavelet DWT + MMSE-LSA (replaces Wiener)
- Created `directives/04_1d_cnn_model.md` — Conv-TasNet waveform model
- Created `directives/05_unet_film_model.md` — U-Net + Self-Attention + FiLM + CRM + MetricGAN+ with HASPI discriminator
- Created `directives/05b_mamba_film_model.md` — Mamba/SSM + FiLM
- Updated `directives/06_evaluation.md` — 5 models, 3 audiogram profiles, HASPI vs STOI comparison strategy
- Updated `AGENTS.md` — phase table with Nov 1 deadline, 5-model architecture summary
- Updated `context.md` — decisions log, file index, environment dependencies, open tasks
- Deleted superseded: `02_wiener_baseline.md`, `04_generic_dnn.md`, `05_film_conditioning.md`
- Created `SESSION_START.md` — copy-paste prompt template for any new chat session
- Replaced `docs/presentation.html` — 14 slides → 20-slide full project proposal
- Restructured `shared_context.md` — separate Jwanil and Namya log sections, backfilled all history
- Committed and pushed all changes to GitHub

**Status:** All documentation current. Directory clean. 9 active directives. SESSION_START.md ready.

**Next action for Jwanil:** Begin Phase 0 — read `directives/00_dsp_fundamentals.md` and run `execution/00_verify_setup.py`.

---

### [2026-08-16 22:20–22:28] — Jwanil — Future Scope and Presentation DOCX

**Actions:**
- Added "Future Scope" slide (Slide 18) to `docs/presentation.html` covering hardware testing, paper submission, Indian language testing, federated learning, and mobile app.
- Created `execution/generate_presentation_docx.py` using `python-docx` to generate a meticulously styled DOCX version of the 21-slide proposal.
- Generated `docs/Project_Proposal_Hearing_Aid_Speech_Enhancement.docx`.
- Committed and pushed to GitHub.

**Status:** Presentation and DOCX proposal document are finalized.

**Next action:** Begin Phase 0.

---

### [2026-08-17 09:20 IST] — Jwanil — Proposal DOCX (report-style design)

**Actions:**
- Created `execution/generate_proposal_docx.py` — new script using the EXACT same design language as `generate_report.py` (white background, `1A56AA` blue headings, `EBF3FB` alternating table rows, `F5F5F5` code blocks, `4A90D9` horizontal rules, Calibri 11pt body).
- Generated `docs/Minor_Project_Proposal_Hearing_Aid_Speech_Enhancement.docx` — 12-section academic proposal document including: Abstract, Problem Statement, Literature Review (with RASTA explanation), Background Theory (7 topics including Mamba/SSM), Methodology (all 5 models with code diagrams), Datasets, Evaluation Metrics, Timeline, Expected Results, Future Scope (6 extensions), Conclusion, References (18 citations).
- Committed and pushed to GitHub.

**Status:** Both `generate_report.py` and `generate_proposal_docx.py` now exist. Run either to regenerate the respective .docx file.

**Next action:** Begin Phase 0.

---

## Open Tasks

- [ ] Install Python dependencies: `pip install torch torchaudio asteroid speechbrain pyclarity pystoi pesq PyWavelets mamba-ssm`
- [ ] Run `execution/00_verify_setup.py` — confirm all imports work
- [ ] Download NOIZEUS dataset (free, small — use for immediate testing)
- [ ] Apply for Clarity Challenge (CEC2/CEC3) dataset access at claritychallenge.org
- [ ] Source TIMIT (check college library for LDC access) or use LibriSpeech as substitute
- [ ] Run `execution/01_stft_visualize.py` — first audio/spectrogram visualization
- [ ] Generate `execution/generate_report.py` — run to create the .docx report

---

## Decisions Made

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-08-11 | Architecture: U-Net + FiLM conditioning | Best novelty-to-feasibility ratio for minor project scope |
| 2026-08-11 | Primary metric: HASPI/HASQI | Hearing-aid-specific — differentiates from generic denoising work |
| 2026-08-11 | Primary dataset: Clarity Challenge | Purpose-built for HA research, includes audiograms |
| 2026-08-11 | Framework: PyTorch + SpeechBrain + Asteroid | Mature, well-documented, pretrained recipes |
| 2026-08-15 | **Replace Wiener filter with Wavelet DWT + MMSE-LSA** | Faculty feedback — more sophisticated classical baselines |
| 2026-08-15 | **Replace Transformer with Mamba/SSM** | Research sweep: Mamba beats Transformer on PESQ (3.69 vs ~3.4) AND is faster (O(T) vs O(T²)) |
| 2026-08-15 | **Add Complex Ratio Mask (CRM)** | Research: phase-aware masking eliminates musical noise artifacts |
| 2026-08-15 | **MetricGAN+ with HASPI discriminator** | Novel: direct HASPI optimisation has not been published — our specific contribution |
| 2026-08-15 | **Add 1D CNN as 3rd model** | Faculty suggestion: waveform-domain approach, natural low-latency comparison |
| 2026-08-15 | **Datasets: TIMIT + NOIZEUS** | Faculty suggestion + standard evaluation benchmarks in SE literature |
| 2026-08-15 | **Deadline: November 1, 2026** | Faculty-set deadline |

---

## Known Issues / Blockers

*None yet.*

---

## Key File Index

| File | Purpose |
|------|---------|
| `AGENTS.md` / `GEMINI.md` / `CLAUDE.md` | Agent operating instructions (read first) |
| `context.md` | This file — session state |
| `shared_context.md` | Cross-partner collaboration log (Jwanil + Namya) |
| `docs/simple_guide.md` | **Plain-English guide — START HERE for any new session** |
| `docs/project_overview.md` | Full project background — feed to new sessions |
| `docs/everything_from_scratch.md` | Deep technical theory explainer |
| `docs/presentation.html` | Faculty pitch deck |
| `directives/00_dsp_fundamentals.md` | Phase 0 SOP |
| `directives/01_audiology.md` | Phase 1 SOP |
| `directives/02_classical_baselines.md` | Phase 2 SOP — Wavelet + MMSE-LSA |
| `directives/03_data_pipeline.md` | Phase 3 SOP |
| `directives/04_1d_cnn_model.md` | Phase 4 SOP — Conv-TasNet |
| `directives/05_unet_film_model.md` | Phase 5 SOP — U-Net + FiLM (Core) |
| `directives/05b_mamba_film_model.md` | Phase 5b SOP — Mamba/SSM + FiLM |
| `directives/06_evaluation.md` | Phase 6 SOP — Full evaluation |
| `directives/07_report.md` | Phase 7 SOP — Report and demo |
| `execution/` | Numbered Python scripts (00–26+) |
