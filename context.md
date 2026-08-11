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
| 2 | Wiener Filter Baseline | ⬜ Not Started | |
| 3 | Data Pipeline | ⬜ Not Started | |
| 4 | Generic DNN Denoiser | ⬜ Not Started | |
| 5 | FiLM Conditioning | ⬜ Not Started | |
| 6 | Full Evaluation | ⬜ Not Started | |
| 7 | Report + Demo | ⬜ Not Started | |

**Legend:** ✅ Done | 🟡 In Progress | ⬜ Not Started | ❌ Blocked

---

## Environment

- **OS:** macOS
- **Python:** (set when installed)
- **CUDA available:** (check when GPU setup)
- **Workspace path:** `/Users/jwanilmodi/Web development/hearing-aid-speech-enhancement`
- **Key dependencies:** torch, torchaudio, speechbrain, pyclarity, pystoi

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

## Open Tasks

- [ ] Install Python dependencies (`pip install torch torchaudio librosa speechbrain pyclarity pystoi`)
- [ ] Run `execution/00_verify_setup.py` — confirm all imports work
- [ ] Download VoiceBank-DEMAND dataset (small, for early experiments)
- [ ] Apply for Clarity Challenge dataset access at claritychallenge.org
- [ ] Run `execution/01_stft_visualize.py` — first audio/spectrogram visualization
- [ ] Present to faculty (presentation.html ready)

---

## Decisions Made

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-08-11 | Architecture: U-Net + FiLM conditioning | Best novelty-to-feasibility ratio for minor project scope |
| 2026-08-11 | Primary metric: HASPI/HASQI | Hearing-aid-specific — differentiates from generic denoising work |
| 2026-08-11 | Primary dataset: Clarity Challenge | Purpose-built for HA research, includes audiograms |
| 2026-08-11 | Baseline: Wiener Filter | Well-understood, gives clear comparison point |
| 2026-08-11 | Framework: PyTorch + SpeechBrain | Mature, well-documented, has pretrained recipes |

---

## Known Issues / Blockers

*None yet.*

---

## Key File Index

| File | Purpose |
|------|---------|
| `AGENTS.md` | Agent operating instructions (read first) |
| `context.md` | This file — session state |
| `shared_context.md` | Cross-partner collaboration log |
| `docs/project_overview.md` | Full project background — feed to new sessions |
| `docs/everything_from_scratch.md` | Theory explainer — all concepts from scratch |
| `docs/presentation.html` | Faculty pitch deck |
| `directives/` | Phase-by-phase SOPs |
| `execution/` | Deterministic Python scripts |
| `src/models/` | Model architecture code |
