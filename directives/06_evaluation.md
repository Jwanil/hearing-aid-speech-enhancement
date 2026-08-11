# Directive 06 — Full Evaluation

**Phase:** 6  
**Goal:** Rigorously compare all three models (Wiener, Generic, Personalized) and compute latency/size metrics.  
**Estimated time:** 1 week  
**Output:** Final numbers for the report and comparison plots.

---

## What to Learn

1. **HASPI / HASQI** — understand exactly how these metrics work (they take the audiogram as input).
2. **STOI / SI-SDR** — standard speech metrics.
3. **Latency profiling** — how to measure inference time in PyTorch.
4. **Parameter counting** — how to count trainable parameters in PyTorch.

---

## Resources (in order)

1. **Reference:** `docs/everything_from_scratch.md` — Part 8 & 9.
2. `pyclarity` documentation for HASPI/HASQI.

---

## Execution Scripts

| Script | What it does |
|--------|-------------|
| `execution/14_evaluate_all.py` | Runs all test data through all three models and computes STOI, HASPI, HASQI, SI-SDR |
| `execution/15_efficiency_metrics.py` | Computes parameters and measures inference time (latency) per frame |
| `execution/16_plot_results.py` | Generates bar charts and scatter plots for the report |

---

## Tasks

- [ ] Write evaluation script to process the test set through all three models.
- [ ] Save the results to a CSV file in `results/`.
- [ ] Run `15_efficiency_metrics.py` to get the latency constraint numbers.
- [ ] Run `16_plot_results.py` to create visual comparisons.
- [ ] Ensure that HASPI shows the largest gap between Generic and Personalized models (this is the core thesis of your project).

---

## Success Criteria

You have hard, reproducible numbers proving that the Personalized model is better than the Generic model for hearing-impaired listeners, along with honest latency numbers.

---

## Learnings Log

*(Agent: append findings here as you work through this phase)*
