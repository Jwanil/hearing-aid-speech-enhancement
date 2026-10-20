# Hearing Aid Speech Enhancement 🎧

> **Audiogram-Personalized, Low-Latency Speech Enhancement for Hearing Aids using Deep Learning**

Minor Project — [Your College Name]

---

## Team
- Jwanil ([@Jwanil](https://github.com/Jwanil)) — Lead Developer
- Namya Shah — Co-developer

---

## Project Overview

Building a deep learning model that personalizes noise reduction for hearing aids based on the user's individual audiogram (hearing loss profile), using **FiLM conditioning** — so the model literally processes speech differently for two different people with different hearing-loss profiles.

### The Problem
466 million people have disabling hearing loss worldwide. Hearing aids amplify sound, but the #1 complaint is understanding speech in noisy environments. Current noise reduction in hearing aids is **generic** — same algorithm for every user. But every person's hearing loss is unique.

### Our Solution
- Take a user's audiogram (6-frequency hearing threshold vector) as a **runtime input**
- Use FiLM conditioning to adapt the denoiser's behavior per person
- Evaluate with HASPI/HASQI — metrics specifically designed for hearing-impaired listeners
- Report latency + model size tradeoffs for real hearing-aid deployment

---

## Project Structure

```
hearing-aid-speech-enhancement/
│
├── docs/                    # Presentation and reports
│   └── presentation.html
│
├── src/                     # Source code (to be created)
│   ├── data/                # Dataset loading and preprocessing
│   ├── models/              # Model architectures
│   ├── train.py             # Training script
│   └── evaluate.py          # Evaluation script
│
├── notebooks/               # Jupyter notebooks for experiments
│
├── results/                 # Saved models, metrics, plots
│
└── README.md
```

---

## Local RTX 4090 setup

```bash
# Install the CUDA-enabled torch/torchaudio build appropriate for the host first.
python -m pip install -r requirements.txt

# Verify the local GPU build.
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA-enabled PyTorch is required"
print(torch.cuda.get_device_name(0), torch.version.cuda)
PY
```

Phase 5 training is local-first. From the repository root, verify the complete
dataset gate (`clean=11071`, `noise=930`, `rir=60000`) and then follow
`docs/training_platform.md` for the CUDA smoke checks and Phase A command.

---

## References

- Diehl et al., "Restoring speech intelligibility for hearing aid users with deep learning", *Nature Scientific Reports* (2023)
- Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer" (2017)
- NeuroAMP: End-to-end Deep Neural Amplifier for Personalized Hearing Aids (arXiv 2025)
- [Clarity Challenge](https://claritychallenge.org)
