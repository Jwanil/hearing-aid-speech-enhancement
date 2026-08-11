# Directive 07 — Report & Demo

**Phase:** 7  
**Goal:** Package the work into a final college minor project report and a compelling audio demo.  
**Estimated time:** 1 week  
**Output:** Final PDF report and a web or local audio demo.

---

## What to Learn

1. **Academic Writing** — how to structure a machine learning paper (Abstract, Intro, Related Work, Method, Results, Conclusion).
2. **Audio normalization** — ensuring your before/after audio samples are at the same volume so the comparison is fair.

---

## Resources

1. Previous college minor project reports (ask seniors for format).
2. The `docs/everything_from_scratch.md` file (you can lift explanations directly from this).

---

## Execution Scripts

| Script | What it does |
|--------|-------------|
| `execution/17_generate_audio_samples.py` | Runs specific noisy files through all 3 models and saves `.wav` files for the demo |
| `execution/18_build_demo_html.py` | (Optional) Creates a simple HTML page to click and listen to before/after comparisons |

---

## Tasks

- [ ] Write `execution/17_generate_audio_samples.py`. Pick 3 distinct audiogram profiles (e.g., normal, severe high-frequency loss, flat loss).
- [ ] For each profile, save: `noisy.wav`, `wiener.wav`, `generic.wav`, `personalized.wav`.
- [ ] Draft the project report.
- [ ] Make sure the report explicitly addresses the **latency** constraint as a limitation/future work (this shows maturity).
- [ ] Prepare for final presentation.

---

## Success Criteria

You have a bound/PDF report and can play an audio file to your faculty that clearly demonstrates the personalized model sounding better *for that specific simulated hearing loss*.

---

## Learnings Log

*(Agent: append findings here as you work through this phase)*
