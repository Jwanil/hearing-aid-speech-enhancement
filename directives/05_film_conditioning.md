# Directive 05 — FiLM Conditioning (The Core Innovation)

**Phase:** 5  
**Goal:** Modify your generic denoiser to take the audiogram as an input and personalize its processing.  
**Estimated time:** 2–3 weeks  
**Output:** The `PersonalizedDenoiser` model, which demonstrably changes its output based on the audiogram.

---

## What to Learn

1. **FiLM (Feature-wise Linear Modulation)** — read the paper.
2. **Conditioning Neural Networks** — how side-information is injected into a main network stream.
3. **Broadcasting in PyTorch** — how to multiply a 1D scale factor across a 2D spatial feature map.

---

## Resources (in order)

1. **Reference:** `docs/everything_from_scratch.md` — Part 6.
2. **Paper:** Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer" (2017).

---

## Execution Scripts

| Script | What it does |
|--------|-------------|
| `execution/10_film_layer.py` | PyTorch module for the FiLM generator and application |
| `execution/11_model_personalized.py` | The updated U-Net that includes the FiLM layer |
| `execution/12_train_personalized.py` | Training loop (updated to pass the audiogram to the model) |
| `execution/13_verify_personalization.py` | Crucial script: feeds the *same* noisy audio but *different* audiograms and verifies the output masks are different |

---

## Tasks

- [ ] Write the `FiLMLayer` in `execution/10_film_layer.py`.
- [ ] Integrate it into the bottleneck or decoder of your U-Net in `execution/11_model_personalized.py`.
- [ ] Retrain the model on the same data, but this time passing the audiogram vectors to the forward pass.
- [ ] **Crucial test:** Run `execution/13_verify_personalization.py`. If the output doesn't change when the audiogram changes, the model has ignored the conditioning (mode collapse). Debug the gradients.

---

## Success Criteria

The model successfully trains, and you can prove that it treats a high-frequency loss audiogram differently than a low-frequency loss audiogram for the exact same input audio.

---

## Learnings Log

*(Agent: append findings here as you work through this phase)*
