# Directive 05b — Mamba / State Space Model (SSM) + FiLM

**Phase:** 5b  
**Goal:** Build the strongest and most latency-efficient deep learning model — a Mamba-based architecture with FiLM audiogram conditioning. This replaces the original Transformer encoder-decoder plan.  
**Estimated time:** 1 week (Oct 19 – Oct 25, 2026)  
**Lead:** Jwanil  
**Output:** A Mamba+FiLM model achieving near-SOTA performance while remaining deployable.

---

## Why Mamba Instead of a Transformer?

This was the biggest architectural decision from the post-meeting research sweep.

| Property | Standard Transformer | Mamba (SSM) | Why It Matters for Hearing Aids |
|---|---|---|---|
| Complexity | O(T²) — quadratic | O(T) — linear | Hearing aids have ~1 MFLOP compute budget |
| Causal mode | Requires future masking | Naturally causal (left-to-right) | Real-time = cannot look at future audio frames |
| Long sequences | Slows exponentially | Constant speed per frame | Hearing aids process audio continuously, forever |
| Best PESQ (VoiceBank-DEMAND) | ~3.40 | **3.69** (SEMamba) | More accurate AND faster |
| Parameters | Often 10-30M | 2-5M | Smaller models fit on edge devices |

**Mamba is not a compromise — it is strictly better than Transformers for real-time audio processing.** This is why we replace the Transformer plan with Mamba.

---

## What is a State Space Model (SSM)?

A State Space Model describes a system with memory: the current output depends on everything that happened before, compressed into a **hidden state** `h`.

**Classic SSM equations:**
```
h(t) = A × h(t-1) + B × x(t)    ← "hidden state update" (what to remember)
y(t) = C × h(t)                  ← "output" (what to predict)

where:
  x(t) = current input (audio frame at time t)
  h(t) = hidden state (compressed memory of all past inputs)
  y(t) = output at time t
  A, B, C = learned matrices
```

**Mamba's key innovation — selective state spaces:**

Standard SSMs have fixed A, B, C matrices (same for every input). Mamba makes A, B, C **input-dependent** — they are computed from the current input `x(t)`:

```
A(t), B(t), C(t) = linear_projection(x(t))   ← computed from input!
h(t) = A(t) × h(t-1) + B(t) × x(t)          ← selective memory update
y(t) = C(t) × h(t)                            ← selective output
```

**What "selective" means in practice:**
- When it hears a speech onset (new phoneme), it learns to "open the gate" and update h strongly
- When it hears steady background noise, it learns to "close the gate" and barely update h
- This selectivity is learned end-to-end from data

**Analogy:** An old RNN is like a conveyor belt that moves everything forward at the same speed. Mamba is like a conveyor belt where a smart gate decides what gets passed forward and what gets discarded, based on what's currently being processed.

---

## Reference Architecture: SEMamba

**SEMamba** (Chao et al., 2024) is the current state-of-the-art speech enhancement model. It replaces the attention layers in typical SE models with Mamba SSM blocks.

**Key results of SEMamba:**
- PESQ: **3.69** (with Perceptual Contrast Stretching / PCS augmentation)
- PESQ: 3.55 (base SEMamba without augmentation)
- ~12% fewer FLOPs than equivalent Transformer
- Works in causal (real-time) mode with minor performance degradation

### SEMamba Architecture

```
Noisy Spectrogram  Y ∈ ℂ^{F × T}
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  Encoder                                              │
│  2D Convolutional front-end                          │
│  Extract initial features: (B, C_enc, F', T')        │
└────────────────────────────┬─────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────┐
│  Mamba Blocks × N (stacked)                          │
│  Each block processes across the TIME dimension:     │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │  Forward Mamba scan (left → right in time)   │   │
│  │  + Backward Mamba scan (right → left)        │   │
│  │  → Bidirectional context without attention   │   │
│  │  + LayerNorm + residual connection           │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  After N blocks:                                     │
│  Apply FiLM conditioning here ← audiogram            │
└────────────────────────────┬─────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────┐
│  Decoder                                             │
│  2D Convolutional back-end                           │
│  → Complex Ratio Mask (CRM): 2 channels             │
└──────────────────────────────────────────────────────┘
                             │
                 Apply CRM to noisy spectrogram
                             │
                           ISTFT
                             │
                    Enhanced audio output
```

**Note on bidirectional Mamba:** In offline (non-real-time) mode, we run Mamba both left-to-right AND right-to-left and combine the results — this gives context from both past and future. In real-time mode, we use only the forward (causal) scan. We will implement both and report the performance difference.

---

## Adding FiLM Conditioning

The FiLM layer is identical to the one in Directive 05 (U-Net model), applied after the Mamba blocks:

```python
class MambaFiLMBlock(nn.Module):
    """Mamba block followed by FiLM conditioning."""
    def __init__(self, d_model, audiogram_dim=6):
        super().__init__()
        self.mamba = MambaLayer(d_model=d_model)   # from mamba_ssm library
        self.norm  = nn.LayerNorm(d_model)
        
        # FiLM generator: audiogram → scale and shift
        self.film_gen = nn.Sequential(
            nn.Linear(audiogram_dim, 64),
            nn.ReLU(),
            nn.Linear(64, d_model * 2)   # → gamma (d_model) + beta (d_model)
        )
    
    def forward(self, x, audiogram):
        # x shape: (B, T, d_model) — batch, time frames, features
        
        # Mamba scan (memory-efficient sequential processing)
        x = self.norm(x + self.mamba(x))
        
        # FiLM personalisation
        film_params = self.film_gen(audiogram)          # (B, d_model × 2)
        gamma = film_params[:, :self.d_model]           # (B, d_model)
        beta  = film_params[:, self.d_model:]           # (B, d_model)
        
        # Broadcast and apply: (B, 1, d_model) × (B, T, d_model) → (B, T, d_model)
        x = gamma.unsqueeze(1) * x + beta.unsqueeze(1)
        
        return x
```

---

## Execution Scripts

| Script | What It Does |
|--------|-------------|
| `execution/19_model_mamba_film.py` | Full Mamba+FiLM model definition |
| `execution/20_train_mamba_film.py` | Training loop (same SI-SDR + MetricGAN+ approach as directive 05) |
| `execution/21_eval_mamba_film.py` | Evaluation on NOIZEUS, compute all metrics |
| `execution/22_causal_vs_noncausal.py` | Compare causal (real-time) vs non-causal (offline) Mamba performance |

---

## Tasks

- [ ] Install the Mamba library: `pip install mamba-ssm` (requires a compatible GPU and CUDA)
  - **If no GPU available:** Use a simplified SSM implementation in pure PyTorch (ask the agent to provide this)
  - **Alternatively:** Clone the SEMamba GitHub repo and adapt their implementation
- [ ] Read the SEMamba paper (Chao et al., 2024) — particularly Sections 3 and 4
- [ ] Define the full model in `execution/19_model_mamba_film.py`
- [ ] Write `execution/20_train_mamba_film.py`:
  - Same structure as the U-Net training (SI-SDR primary + MetricGAN+ HASPI discriminator)
  - Start with a small model (N=4 Mamba blocks) to verify training before scaling up
- [ ] Train and evaluate
- [ ] Run `execution/22_causal_vs_noncausal.py`:
  - Causal mode = real-time hearing aid deployment (forward scan only)
  - Non-causal mode = offline processing (forward + backward scan)
  - Report the performance gap — this quantifies the cost of real-time constraint
- [ ] Re-run the personalisation verification from Directive 05: different audiograms must produce different masks

---

## Resources

- **Paper:** Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023) — read Sections 1-3
- **Blog:** "The Annotated Mamba" — Google this; provides line-by-line Python walkthrough
- **Paper:** Chao et al., "SEMamba" (2024) — the speech enhancement application we are adapting
- **Code:** SEMamba GitHub — https://github.com/RoyChao19477/SEMamba
- **Library:** mamba-ssm — https://github.com/state-spaces/mamba — official Mamba PyTorch implementation
- **Video:** YouTube — "Mamba SSM explained" — several good walkthroughs (~20-30 min)
- **Video:** YouTube — "State Space Models explained" — for the mathematical foundation

---

## Success Criteria

1. Mamba+FiLM achieves higher PESQ and HASPI than the U-Net+FiLM (Model 4)
2. Causal Mamba latency is measurably lower than Transformer would be (measure with torch.profiler)
3. The personalisation verification test passes (different audiograms → different outputs)
4. The causal vs non-causal comparison is documented and reported honestly

---

## Learnings Log

*(Agent: append findings here — GPU requirements, mamba-ssm installation issues, whether causal mode significantly hurts performance)*
