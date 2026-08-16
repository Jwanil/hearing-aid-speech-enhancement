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

> ⚠️ **Namya — before your first entry:** Pull from GitHub, read `SESSION_START.md`, then read `docs/simple_guide.md`. Those two files will give you everything you need to start.

---

*(No entries yet — Namya has not started a session.)*

---
