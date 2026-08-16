# SESSION START — Hearing Aid Speech Enhancement Project

> **Copy-paste this entire block at the start of every new chat session** to give the AI full project context instantly.

---

## The Prompt (Copy Everything Below This Line)

---

I'm continuing work on my college minor project: **Audiogram-Personalised Speech Enhancement for Hearing Aids using Deep Learning**.

**Team:** Jwanil Modi (23BIT194) and Namya Shah (23BIT027)  
**Deadline:** November 1, 2026

Please read these files in order to get full context before we start. They are all inside the project directory at `/Users/jwanilmodi/Web development/hearing-aid-speech-enhancement/`:

1. `@AGENTS.md` — your operating instructions, the phase table, and what files exist  
2. `@context.md` — current phase status, open tasks, decisions log  
3. `@shared_context.md` — collaboration notes between Jwanil and Namya  
4. `@docs/simple_guide.md` — plain-English guide: what we're building, all concepts defined simply, resources per phase, the full architecture, and the timeline  
5. `@docs/project_overview.md` — full project background document

After reading all 5 files, please confirm:
- **What phase are we in?**
- **What is the exact next task?**
- **Are there any blockers or open questions I should know about?**

Then wait for my instructions.

---

## What This Project Is (30-Second Summary for Context)

We are building a speech enhancement system for hearing aids that **personalises noise removal based on each person's audiogram** (hearing test result). Unlike commercial hearing aids that use the same noise-reduction algorithm for everyone, our system adjusts how it cleans speech specifically for each person's hearing loss profile.

**5-model comparison pipeline:**
1. Wavelet DWT (Discrete Wavelet Transform) denoising — classical baseline
2. MMSE-LSA (Minimum Mean Square Error – Log Spectral Amplitude) filter — industry standard hearing aid algorithm
3. 1D CNN (Conv-TasNet) — waveform-domain deep learning, no audiogram
4. U-Net + Self-Attention + FiLM — our core contribution (personalised, complex masking, MetricGAN+ training)
5. Mamba/SSM (State Space Model) + FiLM — current SOTA (State of the Art), linear complexity

**Key evaluation metrics:** HASPI (Hearing Aid Speech Perception Index), HASQI (Hearing Aid Speech Quality Index), STOI (Short-Time Objective Intelligibility), SI-SDR (Scale-Invariant Signal-to-Distortion Ratio), latency (must be under 10 milliseconds)

**Datasets:** TIMIT, NOIZEUS, Clarity Challenge CEC2/CEC3, VoiceBank-DEMAND, MUSAN (for noise augmentation)

---

## Key Architecture Decisions (Already Made — Do Not Re-Question)

| Decision | What We Chose | Why |
|---|---|---|
| Classical baselines | Wavelet DWT + MMSE-LSA | Faculty feedback (Wiener too basic) |
| Phase-aware masking | Complex Ratio Mask (CRM) | Eliminates musical noise artifacts |
| Training loss | SI-SDR + MetricGAN+ with HASPI discriminator | Novel: HASPI-optimised GAN not published before |
| Strongest DL model | Mamba/SSM (not Transformer) | PESQ 3.69 vs ~3.4 Transformer, linear complexity O(T) |
| Conditioning method | FiLM (Feature-wise Linear Modulation) | Proven method for side-input conditioning |
| Datasets | TIMIT + NOIZEUS + Clarity + VoiceBank | Faculty-approved |

---

## Phase Timeline

| Phase | Description | Dates | Status |
|---|---|---|---|
| 0 | DSP Fundamentals | Aug 15–22 | 🟡 In Progress |
| 1 | Audiology + Audiograms | Aug 23–29 | ⬜ |
| 2 | Classical Baselines (Wavelet + MMSE-LSA) | Aug 30–Sep 6 | ⬜ |
| 3 | Data Pipeline | Sep 7–20 | ⬜ |
| 4 | 1D CNN (Conv-TasNet) | Sep 21–Oct 4 | ⬜ |
| 5 | U-Net + Attention + FiLM | Oct 5–18 | ⬜ |
| 5b | Mamba/SSM + FiLM | Oct 19–25 | ⬜ |
| 6 | Full Evaluation | Oct 26–28 | ⬜ |
| 7 | Report + Audio Demo | Oct 29–31 | ⬜ |
| 🎯 | **SUBMIT** | **Nov 1, 2026** | |

---

## Important Notes for the AI Agent

- **Always give full forms** of all abbreviations the first time you use them (e.g., STFT = Short-Time Fourier Transform)
- **Explain complex terms simply** — Jwanil and Namya are undergrads, not PhDs
- **Read the phase directive first** before starting any implementation work: `directives/0X_PHASE_NAME.md`
- **Log all significant actions** to `context.md` after completing them
- **Do not re-discuss decisions** in the table above — they are final
- **The presentation is at** `docs/presentation.html` — open with Live Server in VS Code
- **The simple guide is the best onboarding document** — recommend it to Namya when she joins a session

---

*This file lives at: `SESSION_START.md` in the project root.*
