# Directive 01 — Audiology & Audiograms

**Phase:** 1  
**Goal:** Understand the domain you are building for. What is hearing loss technically, and how do we represent it?  
**Estimated time:** 3–4 days  
**Output:** Ability to generate synthetic audiograms and simulate hearing loss.

---

## What to Learn

1. **Audiogram basics** — frequency vs dB HL (Hearing Level), normal vs impaired.
2. **Types of hearing loss** — specifically sensorineural loss (hair cell damage).
3. **NAL-R / DSL formulas** — how real hearing aids turn an audiogram into a gain curve.
4. **Hearing loss simulation** — how to process clean speech to sound like a hearing-impaired person hears it.

---

## Resources (in order)

1. **Reference:** `docs/everything_from_scratch.md` — Part 3
2. **Library docs:** `pyclarity` hearing loss simulation tutorial
3. **General knowledge:** Look up "pure-tone audiometry" basics online.

---

## Execution Scripts

| Script | What it does |
|--------|-------------|
| `execution/02_audiogram_generator.py` | Generate synthetic audiograms based on standard clinical distributions (flat, sloping, etc.) |
| `execution/03_simulate_hearing_loss.py` | Use `pyclarity` to pass clean speech through an audiogram and output what it sounds like |

---

## Tasks

- [ ] Read everything about audiograms in the project docs.
- [ ] Understand the 6 standard frequencies: 250, 500, 1000, 2000, 4000, 8000 Hz.
- [ ] Write `execution/02_audiogram_generator.py` to create random realistic vectors.
- [ ] Write `execution/03_simulate_hearing_loss.py` using `pyclarity`.
- [ ] **Crucial:** Listen to clean speech vs simulated hearing loss. If you don't hear the loss of high-frequency consonants, check your code.

---

## Success Criteria

You can take a `.wav` file and an audiogram vector `[10, 15, 30, 50, 65, 80]` and output a new `.wav` file that simulates that specific hearing loss.

---

## Learnings Log

*(Agent: append findings here as you work through this phase)*
