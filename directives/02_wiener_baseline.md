# Directive 02 — Wiener Filter Baseline

**Phase:** 2  
**Goal:** Implement the classical baseline model. You cannot claim your ML model is good unless you can beat a well-tuned classical approach.  
**Estimated time:** 1 week  
**Output:** A working Wiener filter implementation and baseline metrics.

---

## What to Learn

1. **Spectral Subtraction** — the simplest approach, understand why it creates artifacts.
2. **Wiener Filtering** — how it computes a gain mask based on signal and noise power estimates.
3. **Stationary vs Non-stationary noise** — why Wiener filters fail in babble noise (like restaurants) but work okay in fan noise.

---

## Resources (in order)

1. **Reference:** `docs/everything_from_scratch.md` — Part 4
2. **Book:** Philipos Loizou, "Speech Enhancement: Theory and Practice" (Chapters on Wiener filtering)
3. **Scipy documentation:** `scipy.signal.wiener` (though writing it on the spectrogram is better).

---

## Execution Scripts

| Script | What it does |
|--------|-------------|
| `execution/04_wiener_filter.py` | Given a noisy `.wav`, output an enhanced `.wav` using a Wiener filter |
| `execution/05_baseline_eval.py` | Run the Wiener filter on a test set and compute STOI/HASPI |

---

## Tasks

- [ ] Write `execution/04_wiener_filter.py` using NumPy/SciPy.
- [ ] Test it on a simple mixture (clean speech + stationary white noise).
- [ ] Test it on a hard mixture (clean speech + babble noise). Hear the difference in performance.
- [ ] Write `execution/05_baseline_eval.py` to get a numerical baseline score.

---

## Success Criteria

You have a documented STOI and HASPI score for the Wiener filter on your dataset. This is the number your ML model must beat.

---

## Learnings Log

*(Agent: append findings here as you work through this phase)*
