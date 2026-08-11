# shared_context.md — Cross-Partner Collaboration Log

> **AGENT: Read this file at the start of every session alongside `context.md`.**
> This file tracks every significant change made by EITHER partner so neither person is working blind.
> After any prompt that results in a codebase change, append an entry here.

---

## Team

| Name | Role | Machine | Antigravity | Git handle |
|------|------|---------|-------------|------------|
| Jwanil | Lead / ML | Mac | Installed | @Jwanil |
| Namya Shah | Co-developer | [OS] | [Pending setup] | @[namya-handle] |

---

## Live Share Status

- **Host:** Jwanil (typically — whoever starts the session)
- **Join link:** *(generated fresh each session — share on WhatsApp/Discord)*
- **Session convention:** Announce in group chat when starting a Live Share session so partner can join

---

## Sync Protocol

Follow this every time you make a change:

```
1. Make your change (with agent help or manually)
2. Append an entry to THIS file (shared_context.md) immediately
3. Commit and push to GitHub
4. Notify partner in group chat: "pushed [what you did] — please pull"
5. Partner pulls before starting their next session
```

**Never work on the same file simultaneously** without coordinating — Live Share handles real-time, but async work needs git discipline.

---

## Collaboration Log

<!-- 
FORMAT FOR EACH ENTRY:
### [YYYY-MM-DD HH:MM IST] | Author: [Name] | Phase: [Phase number]
**What changed:** Brief description
**Files touched:** list of files
**Agent used:** Antigravity / Claude / Cursor / etc.
**Status after:** what state things are in now
**Action needed from partner:** (if any)
-->

---

### [2026-08-11 21:33 IST] | Author: Jwanil | Phase: Setup

**What changed:** Full project scaffold created from scratch via Antigravity conversation.

**Files created:**
- `README.md` — project overview
- `.gitignore` — ML-appropriate ignore rules
- `AGENTS.md`, `GEMINI.md`, `CLAUDE.md` — agent instruction mirrors
- `context.md` — per-session task log
- `shared_context.md` — this file
- `docs/project_overview.md` — comprehensive project brief
- `docs/presentation.html` — faculty pitch deck (14 slides, premium design)
- `docs/everything_from_scratch.md` — full theory explainer (audio → FiLM → evaluation)
- `directives/00_dsp_fundamentals.md` through `directives/07_report.md` — all phase SOPs
- `execution/00_verify_setup.py` — dependency check script
- `FRIEND_SETUP.md` — onboarding guide for partner

**Agent used:** Antigravity (Gemini)

**Background:** This project came from a Claude conversation that established the project idea. Key decisions: U-Net + FiLM conditioning architecture, HASPI/HASQI evaluation, Clarity Challenge dataset.

**Status after:** Project structure 100% ready. No code written yet.

**Action needed from Namya:**
1. Read `FRIEND_SETUP.md` completely
2. Install Antigravity + VS Code + Live Share + **Five Server** extension
3. Clone the GitHub repo: `https://github.com/Jwanil/hearing-aid-speech-enhancement`
4. Run `execution/00_verify_setup.py` to confirm your environment works
5. Update the Team table above with your OS and GitHub handle

---

## Conflict Resolution

If you and your partner both modified the same file:
1. Don't panic — git merge will catch it
2. Manually resolve the conflict by reading both versions
3. Ask the agent: "Here are two versions of [file], help me merge them keeping both contributions"
4. Log the resolution here

---

## Partner Responsibilities

Track who owns what to avoid duplication:

| Task | Owner | Status |
|------|-------|--------|
| Project setup | Jwanil | ✅ Done |
| Faculty presentation | Jwanil | ✅ Done |
| GitHub repo creation | Jwanil | ⬜ TODO |
| Dataset download (VoiceBank-DEMAND) | TBD | ⬜ TODO |
| Clarity Challenge access request | TBD | ⬜ TODO |
| Wiener Filter baseline | TBD | ⬜ TODO |
| Data pipeline | TBD | ⬜ TODO |
| Generic DNN model | TBD | ⬜ TODO |
| FiLM conditioning | TBD | ⬜ TODO |
| Evaluation scripts | TBD | ⬜ TODO |
| Final report | Both | ⬜ TODO |

---

## Shared Decisions Log

| Date | Author | Decision | Approved by |
|------|--------|----------|-------------|
| 2026-08-11 | Jwanil | Architecture: U-Net + FiLM | Pending partner review |
| 2026-08-11 | Jwanil | Dataset: Clarity Challenge + VoiceBank | Pending partner review |
| 2026-08-11 | Jwanil | Metrics: HASPI, HASQI, STOI, SI-SDR | Pending partner review |
| 2026-08-11 | Jwanil | Framework: PyTorch + SpeechBrain + pyclarity | Pending partner review |
