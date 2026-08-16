# SESSION_START.md — How to Start a New Chat Session

> **Copy the prompt block below** and paste it at the start of any new AI chat session (Antigravity, Claude, or any other tool) to instantly restore full project context.

---

## ✂️ THE PROMPT — Copy everything between the lines

---

I'm continuing work on my college minor project: **Audiogram-Personalised Speech Enhancement for Hearing Aids using Deep Learning**.

**Team:** Jwanil Modi (23BIT194) and Namya Shah (23BIT027) · **Deadline:** November 1, 2026

Please read these files in order before we start:

1. `@AGENTS.md` — your operating instructions, phase table, file index
2. `@context.md` — current phase status, session log, open tasks
3. `@shared_context.md` — collaboration log (separate sections for Jwanil and Namya)
4. `@docs/simple_guide.md` — plain-English guide: concepts, architecture, resources per phase, timeline

After reading, confirm:
- **Which phase are we in?**
- **What is the exact next task?**
- **Any blockers or open questions?**

Then wait for my instructions before doing anything.

---

## ⚠️ Rules for the AI Agent (Always Follow)

1. **Always give full forms** of every abbreviation the first time you use it. Example: STFT (Short-Time Fourier Transform), HASPI (Hearing Aid Speech Perception Index).

2. **Explain complex terms simply.** Jwanil and Namya are undergraduate students, not researchers. Use analogies.

3. **Log every significant action to `context.md`** immediately after completing it. No exceptions. Format:
   ```
   ### [YYYY-MM-DD HH:MM] — [Name] — [Action summary]
   **Actions:** list what was done
   **Status:** state after the action
   **Next action:** what comes next
   ```

4. **Log to `shared_context.md` under the CORRECT person's section.**
   - If Jwanil is running the session → entries go under `## 📘 JWANIL'S SESSION LOG`
   - If Namya is running the session → entries go under `## 📗 NAMYA'S SESSION LOG`
   - **NEVER mix them into one section. This is the most important file in the project.**

5. **Do not re-discuss final decisions.** The following are decided and closed:
   - ✅ Architecture: Wavelet DWT + MMSE-LSA (classical) + 1D CNN + U-Net+FiLM + Mamba+FiLM
   - ✅ Replace Wiener filter with Wavelet DWT + MMSE-LSA
   - ✅ Replace Transformer with Mamba/SSM (PESQ 3.69, linear complexity)
   - ✅ Complex Ratio Mask (CRM) instead of Ideal Ratio Mask (IRM)
   - ✅ MetricGAN+ discriminator trained on HASPI (not PESQ)
   - ✅ Datasets: TIMIT + NOIZEUS + Clarity CEC2/3 + VoiceBank-DEMAND + MUSAN
   - ✅ Deadline: November 1, 2026

6. **Read the phase directive before implementing anything.** The directive files are at `directives/0X_name.md`. They contain the full task breakdown, code structure, and success criteria.

7. **Do not discuss the presentation.** `docs/presentation.html` is for the faculty. Open it with Live Server (Five Server) in VS Code. It does not need to be part of working sessions.

---

## 📌 Quick Reference

| What you need | Where to find it |
|---|---|
| What phase and next task | `context.md` → Current Project Phase |
| Everything from scratch in simple terms | `docs/simple_guide.md` |
| Deep technical background | `docs/everything_from_scratch.md` |
| Phase task breakdown | `directives/0X_phasename.md` |
| My partner's recent work | `shared_context.md` → their log section |
| All decisions (final) | `shared_context.md` → Shared Decisions Log |
| Install command | `pip install torch torchaudio asteroid speechbrain pyclarity pystoi pesq PyWavelets mamba-ssm` |

---

## 📁 Active Directive Files (as of Aug 16, 2026)

| Phase | File | Dates |
|---|---|---|
| 0 | `directives/00_dsp_fundamentals.md` | Aug 15–22 |
| 1 | `directives/01_audiology.md` | Aug 23–29 |
| 2 | `directives/02_classical_baselines.md` | Aug 30–Sep 6 |
| 3 | `directives/03_data_pipeline.md` | Sep 7–20 |
| 4 | `directives/04_1d_cnn_model.md` | Sep 21–Oct 4 |
| 5 | `directives/05_unet_film_model.md` | Oct 5–18 |
| 5b | `directives/05b_mamba_film_model.md` | Oct 19–25 |
| 6 | `directives/06_evaluation.md` | Oct 26–28 |
| 7 | `directives/07_report.md` | Oct 29–31 |
| 🎯 | **SUBMIT** | **Nov 1, 2026** |

---

*This file lives at: `SESSION_START.md` in the project root.*
*Last updated: 2026-08-16 by Jwanil.*
