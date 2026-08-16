# Directive 04 — 1D Convolutional Neural Network (Conv-TasNet Style)

**Phase:** 4  
**Goal:** Build the first deep learning model — a lightweight 1D Convolutional Neural Network (CNN) that operates directly on raw audio waveforms instead of spectrograms.  
**Estimated time:** 2 weeks (Sep 21 – Oct 4, 2026)  
**Lead:** Both  
**Output:** A trained 1D CNN model with documented HASPI, STOI, and SI-SDR scores beating both classical baselines.

---

## What This Model Does (Simple Explanation)

All previous approaches (Wavelet, MMSE-LSA, and our later U-Net) work on **spectrograms** — the 2D "photo of sound" created by Short-Time Fourier Transform (STFT).

This model skips the spectrogram entirely and works on the **raw waveform** — the list of audio samples directly. It learns its own internal frequency decomposition from data instead of using a hand-crafted STFT.

**Why this matters:**
- No STFT window delay → naturally lower latency than spectrogram models
- The model can learn a frequency decomposition better suited to speech than a fixed STFT
- Represents the "waveform approach" in our comparison vs. the "spectrogram approach" of Models 4 and 5

---

## Architecture Reference: Conv-TasNet

**Conv-TasNet** stands for **Convolution-based Time-domain Audio Separation Network** (Luo & Mesgarani, 2019). Despite being designed for source separation, it is the most important and widely-used waveform-domain speech enhancement architecture.

### Pipeline

```
Raw waveform x(t)  ∈  ℝᴸ   (L = number of audio samples)
        │
        ▼
┌────────────────────────────────────────────────────────┐
│  Learned Encoder (replaces STFT)                       │
│  1D Convolution: kernel=16, stride=8                   │
│  Output: N=256 filters × T' time frames               │
│  (like N frequency bins, but learned from data)         │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│  Temporal Convolutional Network (TCN) — The Separator  │
│  8 stacked blocks, each block = depthwise-separable    │
│  1D convolutions with increasing dilation:             │
│    Block 1: dilation = 1   (sees 3 frames at once)    │
│    Block 2: dilation = 2   (sees 5 frames apart)      │
│    Block 3: dilation = 4   (sees 9 frames apart)      │
│    Block 4: dilation = 8   ...                         │
│    Block 5: dilation = 16                              │
│    Block 6: dilation = 32                              │
│    Block 7: dilation = 64                              │
│    Block 8: dilation = 128 (sees 257 frames apart)    │
│  Purpose: captures both short phonemes AND long        │
│  vowels without RNNs (Recurrent Neural Networks)       │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│  Mask Estimation                                        │
│  1×1 Convolution → Sigmoid activation                  │
│  Output: mask M ∈ [0,1]^{N × T'}                      │
└────────────────────────┬───────────────────────────────┘
                         │
                   mask × encoded features
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│  Learned Decoder (replaces ISTFT)                      │
│  1D Transposed Convolution                             │
│  Output: reconstructed waveform ŝ(t)                  │
└────────────────────────────────────────────────────────┘
```

### Why Dilated Convolutions?

A standard convolution with kernel size 3 can only see 3 consecutive frames at once. To see context 1 second into the past, you'd need to stack 100+ layers.

**Dilated convolutions** solve this by skipping frames:
- Dilation 1: looks at positions [t-1, t, t+1] — 3 adjacent frames
- Dilation 2: looks at positions [t-2, t, t+2] — 3 frames, 2 apart
- Dilation 8: looks at positions [t-8, t, t+8] — very spread out

Stacking blocks with exponentially increasing dilation lets the model see a **very large context window** with a reasonable number of parameters and operations.

---

## Implementation Using Asteroid

**Asteroid** is a Python library for source separation and enhancement with production-ready implementations.

```bash
pip install asteroid
```

```python
from asteroid.models import ConvTasNet

model = ConvTasNet(
    n_src=1,         # 1 output source (enhancement, not separation)
    n_filters=256,   # N: number of encoder filters
    n_repeats=3,     # number of TCN repeat cycles
    bn_chan=128,     # bottleneck channels
    hid_chan=256,    # hidden channels in depthwise conv
    skip_chan=128,   # skip connection channels
    kernel_size=16,  # encoder/decoder kernel size
    stride=8         # encoder/decoder stride (controls latency)
)
```

**Note on latency:** The stride parameter directly controls minimum latency. Stride=8 at 16kHz = 0.5ms per step. The total latency also depends on the number of layers and their receptive field.

---

## Execution Scripts

| Script | What It Does |
|--------|-------------|
| `execution/10_model_1d_cnn.py` | Defines the Conv-TasNet model (using Asteroid) |
| `execution/11_train_1d_cnn.py` | Training loop: load data, forward pass, SI-SDR loss, backward pass, optimiser step |
| `execution/12_eval_1d_cnn.py` | Run trained model on NOIZEUS test set, compute all metrics |

---

## Tasks

- [ ] Install Asteroid: `pip install asteroid`
- [ ] Define the model in `execution/10_model_1d_cnn.py`:
  - Import ConvTasNet from Asteroid
  - Set hyperparameters (start with the defaults above)
  - Test that a forward pass works: `model(torch.randn(1, 1, 16000))` should return shape `(1, 1, 16000)`
- [ ] Write `execution/11_train_1d_cnn.py`:
  - Load data from the PyTorch DataLoader (from Phase 3)
  - Use SI-SDR as the loss function (available in Asteroid: `from asteroid.losses import pairwise_neg_sisdr`)
  - Use Adam optimiser, learning rate 3e-4
  - Log training loss to TensorBoard
  - Save checkpoint every 5 epochs to `results/checkpoints/1d_cnn_epoch_{N}.pt`
- [ ] Train for 50 epochs on a small data subset first (20 utterances) — loss should decrease
- [ ] Train fully on the full dataset
- [ ] Write `execution/12_eval_1d_cnn.py`:
  - Load best checkpoint
  - Run on NOIZEUS test set
  - Compute HASPI (using pyclarity, with synthetic audiogram a=[10,15,20,45,70,85])
  - Compute STOI (using pystoi)
  - Compute SI-SDR (using Asteroid)
  - Save to `results/model_1d_cnn_metrics.csv`
- [ ] **Sanity check:** Listen to output on a babble-noise clip. Does it sound better than the MMSE-LSA baseline?

---

## Resources

- **Paper:** Luo & Mesgarani, "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation", IEEE/ACM TASLP (2019) — read Sections 1-3
- **Code:** Asteroid GitHub — https://github.com/asteroid-team/asteroid — look at the Conv-TasNet recipe
- **Code:** Asteroid documentation — https://asteroid-team.github.io/asteroid/
- **Video:** YouTube — "Conv-TasNet explained" — search for this, ~20 min walkthrough
- **Video:** YouTube — "Dilated Convolutions explained" — understand why dilation works

---

## Success Criteria

Your trained 1D CNN:
1. Has lower SI-SDR loss than both classical baselines on the NOIZEUS test set
2. Has higher STOI than both classical baselines
3. Shows some HASPI improvement (even if modest — it has no audiogram knowledge)
4. Can process audio in real-time (measure latency with `torch.profiler`)

---

## Learnings Log

*(Agent: append findings here — note which hyperparameters worked, training time, any instability)*
