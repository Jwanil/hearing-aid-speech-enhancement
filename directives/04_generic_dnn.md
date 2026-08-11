# Directive 04 — Generic DNN Denoiser

**Phase:** 4  
**Goal:** Build a deep learning model that removes noise (without personalization). This is your ML baseline.  
**Estimated time:** 2–3 weeks  
**Output:** A trained U-Net or Conv-TasNet model that predicts an Ideal Ratio Mask (IRM) and beats the Wiener filter.

---

## What to Learn

1. **U-Net Architecture** — encoders, decoders, skip connections.
2. **Masking** — predicting a mask (values 0 to 1) instead of predicting the clean spectrogram directly.
3. **Loss functions for audio** — MSE on spectrograms vs SI-SDR on waveforms.

---

## Resources (in order)

1. **Reference:** `docs/everything_from_scratch.md` — Part 5.
2. Search for basic PyTorch U-Net implementations (you can adapt image U-Nets by changing the channel dimensions).
3. SpeechBrain / Asteroid documentation for reference architectures.

---

## Execution Scripts

| Script | What it does |
|--------|-------------|
| `execution/08_model_generic.py` | PyTorch module defining the U-Net |
| `execution/09_train_generic.py` | Training loop (forward pass, loss, backward pass, optimizer) |

---

## Tasks

- [ ] Define the `GenericDenoiser` class in `execution/08_model_generic.py`.
- [ ] Ensure the output of the model is passed through a Sigmoid activation (so the mask values are strictly between 0 and 1).
- [ ] Write the training loop in `execution/09_train_generic.py`.
- [ ] Train for a few epochs on a small subset to ensure loss decreases.
- [ ] Train fully on the dataset.
- [ ] Evaluate against the Wiener filter using STOI/HASPI.

---

## Success Criteria

The Generic DNN clearly outperforms the Wiener filter on objective metrics, especially in non-stationary noise.

---

## Learnings Log

*(Agent: append findings here as you work through this phase)*
