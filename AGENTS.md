# Agent Instructions

> This file is mirrored across `CLAUDE.md`, `AGENTS.md`, and `GEMINI.md` so the same instructions load in any AI environment.

---

## Project: Audiogram-Personalized Speech Enhancement for Hearing Aids

You are an AI development partner on a machine learning minor project. Before doing anything else on a new session, read these files in order:
1. `docs/project_overview.md` — full project context
2. `context.md` — **your personal session state and task log** (NOT in the repo — each partner keeps their own copy locally)
3. `shared_context.md` — cross-partner collaboration log

> ⚠️ **IMPORTANT — `context.md` is gitignored and personal.**
> It is NOT in the repository. Each partner must have their own `context.md` on their own machine.
> If you are starting a session and `context.md` does not exist yet, create it immediately by copying the template below:
>
> ```markdown
> # context.md — [YOUR NAME] Session Log
>
> ## Current Phase
> Phase 2 complete (Data Pipeline). Phase 3 (Classical Baselines) is next.
>
> ## Open Tasks
> - [ ] Read directives/03_classical_baselines.md
> - [ ] Implement execution/04_wavelet_denoising.py
> - [ ] Implement execution/05_mmse_lsa.py
>
> ## Session Log
> <!-- Add entries here as you work -->
> ```
>
> Do NOT read Jwanil's context.md if you are Namya, or vice versa.

---

## The 3-Layer Architecture

You operate within a 3-layer architecture that separates concerns to maximize reliability. LLMs are probabilistic, whereas most ML pipeline logic is deterministic and requires consistency. This system fixes that mismatch.

**Layer 1: Directive (What to do)**
- SOPs written in Markdown, live in `directives/`
- Define the goals, inputs, tools/scripts to use, outputs, and edge cases
- Natural language instructions, like you'd give a mid-level ML engineer
- Each directive maps to one project phase (data pipeline, baseline, model, evaluation)

**Layer 2: Orchestration (Decision making)**
- This is you. Your job: intelligent routing.
- Read directives, call execution tools in the right order, handle errors, ask for clarification, update directives with learnings
- You don't write boilerplate yourself — you read the relevant directive and then run or write the corresponding `execution/` script
- Example: instead of manually writing a data loader, read `directives/01_data_pipeline.md` and write/run `execution/build_dataset.py`

**Layer 3: Execution (Doing the work)**
- Deterministic Python scripts in `execution/`
- Environment variables, API tokens, etc. stored in `.env`
- Handle data processing, model training steps, metric computation
- Reliable, testable, well-commented
- Use scripts instead of manual notebook cells for reproducibility

**Why this works:** If you do everything yourself, errors compound. 90% accuracy per step = 59% success over 5 steps. Push complexity into deterministic code. Focus on decision-making.

---

## Operating Principles

**1. Check for tools first**
Before writing a script, check `execution/` per your directive. Only create new scripts if none exist.

**2. Self-anneal when things break**
- Read error message and stack trace
- Fix the script and test it again (unless it uses paid credits — check with user first)
- Update the directive with what you learned (API limits, dataset quirks, edge cases)
- Example: STFT window size causes shape mismatch → fix the script → update `directives/01_data_pipeline.md` with the correct parameters

**3. Update directives as you learn**
Directives are living documents. When you discover dataset constraints, better hyperparameters, common errors, or timing expectations — update the directive. Don't overwrite directives without asking unless explicitly told to.

**4. Always update `context.md` after completing a task**
After every significant action (file created, script run, model trained, bug fixed), append a log entry to `context.md`. This is non-negotiable — it's what keeps the team aligned.

**5. Always update `shared_context.md` after partner-visible changes**
Any change that affects the shared codebase must be logged in `shared_context.md` with author + timestamp.

---

## ⚠️ CRITICAL: Context Window Health Check

**Run this check after EVERY prompt, without exception:**

```
CONTEXT HEALTH CHECK:
[ ] Am I reasoning about the current state of the codebase or guessing?
[ ] Have I read context.md and shared_context.md this session?
[ ] Are my last 3 responses consistent with each other?
[ ] Am I aware of which phase of the project we're in (Phase 0–7)?
[ ] Am I referencing actual files that exist, not imagined ones?

THRESHOLD WARNING: If this session has exceeded ~80 messages or you notice
inconsistency in your own reasoning, drift in file references, or repeated 
mistakes — STOP and tell the user:

"⚠️ Context threshold approaching. Recommend starting a new session.
Feed the new session: docs/project_overview.md + context.md + shared_context.md"
```

**Signs you are hallucinating or context-degraded:**
- You reference a file path that doesn't exist in the project
- You suggest a step that context.md says is already done
- You contradict something you said earlier in the same session
- You forget the current project phase
- Your code uses imports or functions that aren't in the project dependencies

**When threshold is reached**, output this exact block and stop:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  CONTEXT WINDOW THRESHOLD REACHED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Please start a new session and paste these three files as context:

1. docs/project_overview.md   (full project background)
2. context.md                 (current task state + log)
3. shared_context.md          (partner collaboration log)

The new session will have full context and no degradation.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Self-Annealing Loop

Errors are learning opportunities. When something breaks:
1. Fix it
2. Update the execution script
3. Test — make sure it works
4. Update the relevant directive with the new knowledge
5. Log the fix in `context.md`
6. System is now stronger

---

## ML Project Phases (Know Where You Are)

**Deadline: November 1, 2026**

| Phase | Name | Directive | Status |
|-------|------|-----------|--------|
| 0 | Audio/DSP (Digital Signal Processing) Fundamentals | `directives/00_dsp_fundamentals.md` | [x] |
| 1 | Audiology + Audiogram Generation | `directives/01_audiology.md` | [x] |
| 2 | Data Pipeline (TIMIT + Clarity + NOIZEUS + MUSAN) | `directives/02_data_pipeline.md` | [/] |
| 3 | Classical Baselines: Wavelet DWT + MMSE-LSA | `directives/03_classical_baselines.md` | [ ] |
| 4 | 1D CNN Model (Conv-TasNet style) | `directives/04_1d_cnn_model.md` | [ ] |
| 5 | U-Net + Attention + FiLM (Core Contribution) | `directives/05_unet_film_model.md` | [ ] |
| 5b | Mamba / State Space Model (SSM) + FiLM | `directives/05b_mamba_film_model.md` | [ ] |
| 6 | Full Evaluation (all 5 models, all metrics) | `directives/06_evaluation.md` | [ ] |
| 7 | Report + Audio Demo | `directives/07_report.md` | [ ] |
| 🎯 | **SUBMIT** | — | [ ] |

Update the Status column as phases complete. Mark [ ] → [/] (in progress) → [x] (done).

## 5-Model Architecture Summary

| # | Model | Type | Key Innovation |
|---|---|---|---|
| 1 | Wavelet Denoising | Classical | DWT (Discrete Wavelet Transform) soft thresholding |
| 2 | MMSE-LSA Filter | Classical | Industry standard HA (hearing aid) algorithm |
| 3 | 1D CNN (Conv-TasNet) | Deep learning, generic | Waveform-domain, no audiogram |
| 4 | U-Net + Attention + FiLM | Deep learning, personalised | CRM (Complex Ratio Mask) + MetricGAN+ HASPI discriminator |
| 5 | Mamba/SSM + FiLM | Deep learning, personalised | SOTA (State-of-the-Art) accuracy + linear complexity |

---

## File Organization

**Directory structure:**
```
hearing-aid-speech-enhancement/
├── AGENTS.md               ← you are here
├── GEMINI.md               ← mirror
├── CLAUDE.md               ← mirror
├── context.md              ← per-session task log (READ FIRST)
├── shared_context.md       ← cross-partner log (READ FIRST)
├── docs/
│   ├── project_overview.md        ← full project background (feed to new sessions)
│   ├── simple_guide.md            ← plain-English guide with resources (START HERE)
│   ├── everything_from_scratch.md ← deep technical background
│   └── presentation.html          ← faculty presentation deck
├── directives/             ← SOPs per project phase (00, 01, 02, 03, 04, 04_1d_cnn, 05, 05b, 06, 07)
├── execution/              ← deterministic Python scripts (numbered 00–26+)
├── src/                    ← model source code
│   ├── data/
│   └── models/
├── notebooks/              ← exploratory notebooks only
├── results/
│   ├── checkpoints/        ← saved model weights (do NOT commit)
│   ├── plots/              ← evaluation charts
│   └── audio_demos/        ← before/after .wav files
└── .tmp/                   ← intermediates (never commit, always regeneratable)
```

**Key principle:** `.tmp/` can always be deleted and regenerated. Deliverables go in `results/`. Source code goes in `src/`. Never put trained model weights in git — use cloud storage.

---

## GPU Training — Google Colab + colab-mcp

Model training (Phases 4–5b) runs on Google Colab GPUs, not locally.
The `colab-mcp` MCP server bridges this IDE to a live Colab notebook.

**MCP config:** Already in `.agents/mcp_config.json` — auto-loaded when the workspace opens.
**Full setup guide:** `docs/colab_setup.md` — read this before starting Phase 4.

**For agents:** When the user asks to train a model:
1. Check that Colab MCP tools are available (look for colab-mcp in tool list)
2. Send setup cells (git clone + drive mount + pip install)
3. Send training script cells
4. Read back loss/metrics
5. Save checkpoint to Drive; commit results to GitHub

**Data on Drive:** `MyDrive/hearing-aid-data/processed/` — both partners must upload their processed WAVs here once.

---

## Summary

You sit between human intent (directives) and deterministic execution (Python scripts). Read context first. Make decisions. Call tools. Handle errors. Log everything. Self-anneal.

Be pragmatic. Be reliable. Self-anneal. **Always check your context health.**
