# Directive 03 — Data Pipeline

**Phase:** 3  
**Goal:** Build the PyTorch Dataset and DataLoader that feeds training data to the model. Garbage in = garbage out.  
**Estimated time:** 1 week  
**Output:** A fast, bug-free data pipeline outputting (noisy_spectrogram, clean_spectrogram, audiogram) triplets.

---

## What to Learn

1. **PyTorch Datasets & DataLoaders** — how `__getitem__` works, collate functions.
2. **On-the-fly mixing** — why you mix speech and noise dynamically during training (infinite augmentation) rather than saving mixed files to disk.
3. **Signal-to-Noise Ratio (SNR)** — how to scale a noise vector to hit a specific target SNR relative to the speech vector.

---

## Resources (in order)

1. PyTorch documentation on `Dataset` and `DataLoader`.
2. SpeechBrain / `torchaudio` tutorials on data loading.

---

## Execution Scripts

| Script | What it does |
|--------|-------------|
| `execution/06_download_data.py` | Script to download VoiceBank-DEMAND or Clarity data to `.tmp/data` |
| `execution/07_data_pipeline.py` | Contains the PyTorch `Dataset` class and testing logic |

---

## Tasks

- [ ] Write `execution/06_download_data.py` to get a small subset of data working locally.
- [ ] Implement `HearingAidDataset` in `execution/07_data_pipeline.py`.
- [ ] For each item, it should:
  1. Load clean speech.
  2. Load a random noise clip.
  3. Mix them at a random SNR (e.g., between -5 dB and +10 dB).
  4. Generate a random synthetic audiogram vector.
  5. Compute the STFT magnitude spectrogram for the noisy and clean speech.
  6. Return `(noisy_mag, clean_mag, audiogram)`.
- [ ] **Crucial test:** Plot the outputs of the DataLoader. Do they look right? Are the shapes matching?

---

## Success Criteria

You can iterate through a DataLoader in a `for` loop, receiving properly batched, augmented tensors on the GPU (or CPU) without memory leaks or crashes.

---

## Learnings Log

*(Agent: append findings here as you work through this phase)*
