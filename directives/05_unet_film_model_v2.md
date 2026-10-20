# Directive 05 — U-Net + Attention + FiLM: Complete Production Architecture Blueprint (Rev. 2.1)

**Phase:** 5 (Core Contribution — Personalized Denoising)  
**Supersedes:** `directives/05_unet_film_model.md` v1.0 (U-Net + Bottleneck Attention + Single-Point FiLM, CRM clamp [-10,10])  
**New Canonical Path:** `directives/05_unet_film_model.md` — also mirrored as `directives/05_unet_film_model_v2.md` & `docs/phase_5_complete_architecture_blueprint.md`  
**Status:** 🟡 Active — Rev 2.1 Conditional GAN Keep  
**Lead:** Both (Jwanil — Architecture/Training, Namya — Pipeline/Evaluation)  
**Deadline:** Oct 18, 2026 (Phase 5), Oct 25, 2026 (Phase 5b Mamba)  
**Hardware Target:** Local NVIDIA RTX 4090 Phase A/B training + Deployable Streaming Variant (`<10 ms` total round-trip, `Cortex-M55 / Ezairo 7150` class)  
**Last Audit:** Adversarial Stage-Gate Audit 2026-10-10 — Findings Incorporated (FiLM Collapse, CRM Explosion, HASPI-GAN Instability, 48 ms Latency Violation, NAL-R Double-Amplification); Rev 2.1 Amendment 2026-10-10 — MetricGAN conditional keep (result > time)

> **Changelog v1 → v2:** Single-point bottleneck FiLM → Per-stage residual FiLM (4 heads). Unbounded Cartesian CRM `[-10,10]` → Bounded polar CRM `|M|≤1` (suppression-only). Pure `-SI-SDR` → Composite `SI-SDR + Multi-Resolution STFT + Compressed Complex + Asymmetric Over-Suppression + Gain Penalty`. Naive 1:1 MetricGAN+ from epoch 0 → Two-phase curriculum (A: composite only 25 epochs, B: stabilized LSGAN fine-tune 10 epochs with spectral norm + R1). Non-causal `n_fft=512` → Causal `n_fft=256, hop=64, center=False` streaming buffer. 6-dim raw audiogram → Continuous `log2-spline → ERB-64 → embedding-128` projection.
> **Changelog v2 → v2.1:** Phase B optional skip → conditional keep via B1/B2 pilots (ΔHASPI ≥+0.015 gate). Result > time: keep MetricGAN+ HASPI even with +5h delay if it improves HASPI/HASQI; delay only on degradation/instability. D discarded at inference — latency (<10ms) unchanged.

---

## Table of Contents

1. [Clinical & Mathematical Foundations](#1-clinical--mathematical-foundations)
2. [Continuous Audiogram Representations](#2-continuous-audiogram-representations)
3. [Complete Causal Architecture — CausalUNetCRM](#3-complete-causal-architecture--causalunetcrm)
4. [Advanced Complex Masking](#4-advanced-complex-masking)
5. [Composite Loss Formulation](#5-composite-loss-formulation)
6. [Data Augmentation & Pipeline Specification](#6-data-augmentation--pipeline-specification)
7. [Two-Phase Training Protocol](#7-two-phase-training-protocol)
8. [Training Stability & Edge Deployment Readiness](#8-training-stability--edge-deployment-readiness)
9. [Streaming Circular Buffer Inference (<10 ms)](#9-streaming-circular-buffer-inference-10-ms)
10. [Four-Stage Ablation & Evaluation Suite](#10-four-stage-ablation--evaluation-suite)
11. [Execution Script Map & Migration Guide](#11-execution-script-map--migration-guide)

---

## 1. Clinical & Mathematical Foundations

### 1.1 Personalized Denoising Problem

Noisy observation at microphone `m(t)`:

```
y(t) = s(t) * h_room(t) + n(t)          // s: clean speech, n: additive noise, h_room: RIR
Y(f,τ) = STFT{y} ∈ ℂ^{F×T} ,  F=129 (n_fft=256) or 257 (n_fft=512 offline), T≈250 per 4s @ hop=64
```

Audiogram `a ∈ ℝ^6` at frequencies `f_a = [250,500,1000,2000,4000,8000] Hz`, values `HL(f) ∈ [0,120] dB HL` normalized `a_norm = a/120 ∈ [0,1]` per `execution/phase2_data_pipeline/03c_data_pipeline.py:87`.

Goal: Learn `F_θ: (Y, a) → M(f,τ) ∈ ℂ^{F×T}` such that `Ŝ = M ⊙ Y` maximizes `HASPI(ISTFT(Ŝ), s_target, a)` while generic `F_θ(Y,0)` does not.

### 1.2 The NAL-R Prescription Boundary — Why Denoising Must Not Amplify

Commercial hearing aids apply prescriptive gain *after* noise reduction (Byrne & Dillon 1986):

```
Pipeline: y → [ NR_θ(a) ] → y_den → [ G_NALR(a) ] → y_nalr → [ WDRC ] → receiver
G_NALR(f) ≈ 0.31·(HL(f)-20) + 0.05·HL(f)   // simplified, 0–35 dB, dominates at 2–4 kHz
```

If `NR_θ` also provides gain `|M(f)| > 1` (as in v1 `clamp[-10,10]` → +20 dB), cascade gain is multiplicative in linear, additive in dB:

```
G_total(f) [dB] = 20·log10|M_FiLM(f)| + G_NALR(f)
For HL(4k)=70 dB → G_NALR≈22 dB. If |M|=2 (+6 dB) → G_total=28 dB → SPL at eardrum ≈93–98 dB →
  exceeds LDL (Loudness Discomfort Level), triggers recruitment (Stevens power law exponent >1),
  and loop gain G_loop(f)=G_total(f)+Venting(f) >0 dB → acoustic feedback howl at 2–4 kHz.
Dead-region (HL>80 dB @ 8kHz) has no IHC → boosting wastes headroom, HASQI collapses.
```

**Enforced Invariant (Non-Negotiable):**

```
Denoiser is suppression-only: |M(f,τ)| ∈ [0,1] ∀ f,τ
Amplification is NAL-R's exclusive responsibility.
Verification: L_gain = mean(ReLU(|M|-1)²) ≡ 0, enforced in loss (§5) and CRM head (§4).
```

Intelligibility vs. quality decoupling: `HASPI` flat + `HASQI` collapse = signature of double-amplification. Evaluation must report both per `directives/06_evaluation.md`.

### 1.3 Hypothesis & Success Criterion

Primary claim (from `docs/project_overview.md:46-48`):

```
ΔHASPI(FiLM vs Generic)  >  ΔSTOI(FiLM vs Generic)   // hearing-specific benefit, not general NR
ΔHASPI_ProfileB > ΔHASPI_ProfileA                     // severe ski-slope benefits most
```

where `Profile A [10,10,10,10,10,10]`, `Profile B [10,15,20,45,70,85]` (presbycusis), `Profile C [60,60,60,60,60,60]`.

---

## 2. Continuous Audiogram Representations

### 2.1 Why Raw 6-Dim Fails

Raw `a ∈ ℝ^6` has 6 DOF. U-Net decoder has `~800k` DOF. Variance `Var(a/120)≈0.08` vs spectrogram variance `~15 dB` → gradient starvation. Audiograms are not independent bins: adjacent thresholds are correlated via `cumsum jitter` model in `execution/phase1_audiology/02_audiogram_generator.py:194`. Need inductive bias matching cochlear frequency warping.

### 2.2 Log-Frequency Cubic Spline Interpolation

Human pitch perception is `log2(f)`. Interpolate 6 points onto dense frequency axis using `PchipInterpolator` (monotonic, avoids spline overshoot):

```python
# frequencies: log2 scale
F_A = np.log2(np.array([250,500,1000,2000,4000,8000]))  # shape (6,)
F_DENSE = np.log2(np.linspace(250,8000,257))           # or 129 for n_fft=256
# HL(f) continuous curve: HL_dense = Pchip(F_A, HL_6)(F_dense) → shape (257,)
# Normalization: hl_dense_norm = HL_dense / 120.0 ∈ [0,1]
```

Benefits: `257`-dim correlates with `F` bins, enables per-frequency FiLM via broadcast; differentiable; captures notches (e.g., 4 kHz noise-induced notch) that 6-point vector smooths away.

### 2.3 ERB Critical-Band Projection (Gammatone)

Human cochlea resolves `~64 ERB` bands at 16 kHz. Project `HL_dense` via gammatone filterbank `W_erb ∈ ℝ^{64×257}` (Slaney):

```
hl_erb = W_erb @ hl_dense   # (64,) — perceptually uniform, decorrelated features
```

`W_erb` is fixed, not learned (or learned with `softmax` row constraint). Reduces 257 → 64, matches `hidden=128` trunk budget, aligns FiLM modulation with auditory filter broadening modeled in `pyclarity` MSBG (`clarity.evaluator.msbg`).

### 2.4 Psychoacoustic Weighting

Optional: Weight `hl_erb` by equal-loudness contour (ISO 226, 60 phon) `w_iso(f)` and by speech importance function `SII(f)` (ANSI S3.5) peaking at `1–4 kHz`. Implement as `hl_weighted = hl_erb * w_sii_erb` where `w_sii_erb` precomputed.

### 2.5 AudiogramEmbedding Module

```python
# execution/phase5_unet_film/20_model_unet_film.py — PART 1
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from scipy.interpolate import PchipInterpolator

class AudiogramEmbedding(nn.Module):
    """6-dim HL → 128-dim continuous embedding via spline + ERB projection."""
    def __init__(self, n_fft=256, sr=16000, n_erb=64, embed_dim=128):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.n_freq = n_fft//2 + 1  # 129 or 257
        self.n_erb = n_erb
        # Fixed log-frequency axes
        self.register_buffer("freqs_dense", torch.linspace(250, 8000, self.n_freq))
        F_A = np.log2(np.array([250,500,1000,2000,4000,8000], dtype=np.float32))
        self.register_buffer("F_A", torch.tensor(F_A))
        # Fixed ERB projection: (64, n_freq) — row-normalized gammatone magnitude
        # Precompute offline via gammatone filterbank or slaney; here random init + softmax for illustration
        # In production: load W_erb.npy from results/phase1_audiology/
        W_erb = torch.randn(n_erb, self.n_freq)
        W_erb = F.softmax(W_erb, dim=-1)  # each ERB band is a soft frequency selector
        self.register_buffer("W_erb", W_erb)  # (64, n_freq), not trainable
        # SII weighting per ERB band (precomputed, shape 64)
        sii = torch.tensor(self._sii_weights(n_erb))  # e.g., peak at mid-freq
        self.register_buffer("sii_w", sii)

        self.mlp = nn.Sequential(
            nn.Linear(n_erb, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        # Learnable scale for spline output
        self.spline_scale = nn.Parameter(torch.tensor(1.0))

    def _sii_weights(self, n_erb):
        # Simplified SII band importance: triangular peaking at ERB ~30 (≈1.5kHz)
        x = np.arange(n_erb)
        w = 1.0 - np.abs(x - n_erb*0.45)/(n_erb*0.45)
        return np.clip(w, 0.2, 1.0).astype(np.float32)

    def forward(self, a):  # a: (B,6) in [0,1] (already /120)
        B = a.shape[0]
        hl_dense = self._interp_log(a)  # (B, n_freq)
        hl_erb = torch.matmul(hl_dense, self.W_erb.T)  # (B,64)
        hl_erb = hl_erb * self.sii_w.unsqueeze(0)  # SII weighting
        emb = self.mlp(hl_erb)  # (B,128)
        return emb, hl_dense

    def _interp_log(self, a):
        a_exp = a.unsqueeze(1)  # (B,1,6)
        hl_dense = F.interpolate(a_exp, size=self.n_freq, mode='linear', align_corners=True).squeeze(1)
        return hl_dense * self.spline_scale  # (B, n_freq)


class FiLMGenerator(nn.Module):
    """Per-stage residual FiLM: embedding(128) → {gamma,beta} per decoder stage, init γ=1, β=0."""
    def __init__(self, embed_dim=128, film_channels=[32,64,128,256]):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(128, 2*c) for c in film_channels])
        for h in self.heads:
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)  # ensures gamma=1, beta=0 at init
        self.channels = film_channels

    def forward(self, emb):  # emb: (B,128)
        h = self.trunk(emb)  # (B,128)
        gammas, betas = [], []
        for head, c in zip(self.heads, self.channels):
            p = head(h)          # (B,2c)
            gamma = 1.0 + p[:, :c]  # residual
            beta  = p[:, c:]
            gamma = torch.clamp(gamma, 0.5, 2.0)
            betas.append(beta)
            gammas.append(gamma)
        return gammas, betas  # lists length 4, each (B,C)

class FiLMLayer(nn.Module):
    def forward(self, feat, gamma, beta):
        # feat: (B,C,H,W), gamma,beta: (B,C)
        return gamma.unsqueeze(-1).unsqueeze(-1) * feat + beta.unsqueeze(-1).unsqueeze(-1)
```

**Implementation note (2026-10-16):** The production implementation uses
explicit log-frequency interpolation because STFT bins are linear in Hz;
`F.interpolate` over bin indices is not equivalent. When loading the
precomputed nonnegative `W_erb.npy`, row-normalize by row sum if necessary and
do **not** apply softmax: softmax would flatten the sparse triangular ERB bands.

---

## 3. Complete Causal Architecture — CausalUNetCRM

### 3.1 Data Flow

```
y(t) @16kHz, a∈ℝ^6
  ↓ causal STFT (n_fft=256, hop=64, win=hann, center=False)  // effective latency 4 ms + 16 ms window tail = 8 ms algorithmic
Y ∈ ℂ^{F×T}, F=129, T≈1000 per 4s  →  |Y|, ∠Y decoupled
  ↓ mag only → Causal Encoder (4 stages, stride 2, depthwise-separable + SE)
Enc1: (1,129,T)→(32,65,T/2)  Enc2→(64,33,T/4)  Enc3→(128,17,T/8)  Enc4→(256,9,T/16)
  ↓ Bottleneck: Causal Attention (causal mask, kernel 3) + FiLM0 (256)
  ↓ Decoder (4 stages, transposed causal conv, per-stage FiLM + skip fusion)
Dec4: Up(256→128)+FiLM1+SE+Concat(Enc3)  Dec3→(64)+FiLM2  Dec2→(32)+FiLM3  Dec1→(2 CRM)
  ↓ Polar CRM head: M_mag∈[0,1], M_phase∈[-π,π]
M(f,τ) = M_mag·exp(j·M_phase)  // |M|≤1
Ŝ = M ⊙ Y  → ISTFT (overlap-add, center=False) → ŝ(t)
```

### 3.2 Causal Encoder — Depthwise Separable + Squeeze-Excitation

```python
class CausalConv2d(nn.Module):
    def __init__(self, cin, cout, k=3, stride=2):
        super().__init__()
        self.pad = (k-1, 0, k-1, 0)
        self.depthwise = nn.Conv2d(cin, cin, k, stride=stride, padding=0, groups=cin, bias=False)
        self.pointwise = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        x = F.pad(x, self.pad)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(self.bn(x))

class SEBlock(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c//r, 1), nn.ReLU(),
            nn.Conv2d(c//r, c, 1), nn.Sigmoid()
        )
    def forward(self, x): return x * self.fc(x)

class CausalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = CausalConv2d(1, 32, stride=2)   # (B,32,65, T/2)
        self.se1 = SEBlock(32)
        self.e2 = CausalConv2d(32, 64, stride=2)  # (B,64,33, T/4)
        self.se2 = SEBlock(64)
        self.e3 = CausalConv2d(64, 128, stride=2) # (B,128,17,T/8)
        self.se3 = SEBlock(128)
        self.e4 = CausalConv2d(128,256, stride=2) # (B,256,9, T/16)
        self.se4 = SEBlock(256)
    def forward(self, x):  # x: (B,1,F,T) magnitude
        s1 = self.se1(self.e1(x))
        s2 = self.se2(self.e2(s1))
        s3 = self.se3(self.e3(s2))
        s4 = self.se4(self.e4(s3))
        return [s1,s2,s3,s4]  # skips
```

### 3.3 Causal Bottleneck Attention

```python
class CausalBottleneckAttention(nn.Module):
    def __init__(self, c=256):
        super().__init__()
        self.q = nn.Conv1d(c, c//8, 1, bias=False)
        self.k = nn.Conv1d(c, c//8, 1, bias=False)
        self.v = nn.Conv1d(c, c, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.ln = nn.LayerNorm(c)
    def forward(self, x):
        B,C,H,W = x.shape
        x_r = x.permute(0,2,1,3).reshape(B*H, C, W)  # (B*H, C, W)
        q = self.q(x_r)
        k = self.k(x_r)
        v = self.v(x_r)
        attn = torch.bmm(q.permute(0,2,1), k) / (C//8)**0.5  # (B*H, W, W)
        mask = torch.triu(torch.ones(W,W, device=x.device), diagonal=1).bool()
        attn.masked_fill_(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(v, attn.permute(0,2,1))  # (B*H,C,W)
        out = out.view(B, H, C, W).permute(0,2,1,3).reshape(B,C,H,W)
        return self.gamma * out + x
```

### 3.4 Per-Stage FiLM Decoder

```python
class CausalDecoderBlock(nn.Module):
    def __init__(self, cin, cout, film_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x, skip, gamma, beta):
        x = self.up(x)
        if skip is not None:
            dh, dw = x.shape[2]-skip.shape[2], x.shape[3]-skip.shape[3]
            if dh>0 or dw>0: x = x[:, :, :skip.shape[2], :skip.shape[3]]
            elif dh<0 or dw<0: x = F.pad(x, (0, max(-dw,0), 0, max(-dh,0)))
            x = gamma.unsqueeze(-1).unsqueeze(-1) * x + beta.unsqueeze(-1).unsqueeze(-1)
            x = self.act(self.bn(x))
            x = torch.cat([x, skip], dim=1)
        else:
            x = gamma.unsqueeze(-1).unsqueeze(-1) * x + beta.unsqueeze(-1).unsqueeze(-1)
            x = self.act(self.bn(x))
        return x
```

### 3.5 Full CausalUNetCRM Assembly

```python
class CausalUNetCRM(nn.Module):
    def __init__(self, n_fft=256, sr=16000):
        super().__init__()
        self.n_fft = n_fft
        self.sr = sr
        self.n_freq = n_fft//2 + 1
        self.a_emb = AudiogramEmbedding(n_fft=n_fft, sr=sr, n_erb=64, embed_dim=128)
        self.film_gen = FiLMGenerator(embed_dim=128, film_channels=[128,64,32,256])
        self.encoder = CausalEncoder()
        self.bottleneck_attn = CausalBottleneckAttention(256)
        self.film_layer = FiLMLayer()

        self.dec4 = CausalDecoderBlock(256, 128, 128)
        self.dec3 = CausalDecoderBlock(256, 64, 64)
        self.dec2 = CausalDecoderBlock(128, 32, 32)
        self.final = nn.Conv2d(64, 2, kernel_size=1)
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, noisy_wav, audiogram):
        B = noisy_wav.shape[0]
        spec = torch.stft(noisy_wav.squeeze(1), n_fft=self.n_fft, hop_length=64,
                          win_length=self.n_fft, window=self.window,
                          center=False, return_complex=True)
        mag = spec.abs().unsqueeze(1)
        phase_noisy = spec.angle()

        emb, _ = self.a_emb(audiogram)
        gammas, betas = self.film_gen(emb)

        skips = self.encoder(mag)
        s1,s2,s3,s4 = skips
        bottleneck = s4
        bottleneck = self.film_layer(bottleneck, gammas[3], betas[3])
        bottleneck = self.bottleneck_attn(bottleneck)

        d4 = self.dec4(bottleneck, s3, gammas[0], betas[0])
        d3 = self.dec3(d4, s2, gammas[1], betas[1])
        d2 = self.dec2(d3, s1, gammas[2], betas[2])
        d2_up = F.interpolate(d2, size=(self.n_freq, spec.shape[-1]), mode='bilinear', align_corners=False)
        logits = self.final(d2_up)

        mag_logit, phase_logit = logits[:,0], logits[:,1]
        m_mag = torch.sigmoid(mag_logit)
        m_phase = torch.tanh(phase_logit) * torch.pi
        m_real = m_mag * torch.cos(m_phase)
        m_imag = m_mag * torch.sin(m_phase)

        spec_real, spec_imag = spec.real, spec.imag
        enh_real = m_real * spec_real - m_imag * spec_imag
        enh_imag = m_real * spec_imag + m_imag * spec_real
        enh_spec = torch.complex(enh_real, enh_imag)

        enh_wav = torch.istft(enh_spec, n_fft=self.n_fft, hop_length=64,
                              win_length=self.n_fft, window=self.window,
                              center=False, length=noisy_wav.shape[-1])
        return enh_wav.unsqueeze(1), (m_real, m_imag, m_mag, m_phase), enh_spec
```

---

## 4. Advanced Complex Masking

### 4.1 Cartesian vs Polar CRM

| Mode | Output | Range | Gradient | Verdict |
|------|--------|-------|----------|---------|
| **Cartesian** (v1) | `M_real, M_imag` direct | `[-10,10]` clamp | Coupled mag/phase, explosion | ❌ Rejected |
| **Polar** (v2) | `M_mag = sigmoid(logit)`, `M_phase = π·tanh(logit)` → `M = M_mag·e^{j·M_phase}` | `|M|∈[0,1]`, `∠M∈[-π,π]` | Decoupled, stable, suppression-only | ✅ Adopted |

### 4.2 Power Compression `p=0.3`

```
|X_c| = |X|^p ,   ∠X_c = ∠X
Compressed complex loss: L_c = MSE(|Ŝ|^p, |S|^p) + λ·MSE(Ŝ_c, S_c)
```

Training uses `p=0.3` (stronger high-frequency lift for fricatives), evaluation reports uncompressed `SI-SDR`.

---

## 5. Composite Loss Formulation

```python
# execution/phase5_unet_film/21_train_unet_film.py — LOSS MODULE
import torch, torch.nn as nn, torch.nn.functional as F
from asteroid.losses import pairwise_neg_sisdr

class CompositeLoss(nn.Module):
    def __init__(self, sr=16000, n_fft_train=256, lambda_sisdr=1.0, lambda_mrstft=0.5,
                 lambda_comp=0.3, lambda_asym=0.5, lambda_gain=0.1, lambda_cons=0.05, p=0.3):
        super().__init__()
        self.sr=sr; self.p=p
        self.l_sisdr=lambda_sisdr; self.l_mrstft=lambda_mrstft
        self.l_comp=lambda_comp; self.l_asym=lambda_asym
        self.l_gain=lambda_gain; self.l_cons=lambda_cons
        self.mrstft_scales=[(256,64),(512,128),(1024,256)]

    def forward(self, est_wav, clean_wav, noisy_spec, m_real, m_imag, m_mag, clean_spec):
        sisdr = pairwise_neg_sisdr(est_wav, clean_wav).mean()

        mrstft=0
        for n_fft, hop in self.mrstft_scales:
            win = torch.hann_window(n_fft, device=est_wav.device)
            s_est = torch.stft(est_wav.squeeze(1), n_fft=n_fft, hop_length=hop, window=win, center=False, return_complex=True)
            s_cln = torch.stft(clean_wav.squeeze(1), n_fft=n_fft, hop_length=hop, window=win, center=False, return_complex=True)
            mag_est, mag_cln = s_est.abs(), s_cln.abs()
            L=min(mag_est.shape[-1], mag_cln.shape[-1])
            mag_est, mag_cln = mag_est[...,:L], mag_cln[...,:L]
            sc = F.mse_loss(mag_est, mag_cln) / (mag_cln.mean()+1e-7)
            mag = F.l1_loss(torch.log(mag_est+1e-5), torch.log(mag_cln+1e-5))
            mrstft+=sc+mag
        mrstft/=len(self.mrstft_scales)

        enh_spec = torch.complex(m_real*noisy_spec.real - m_imag*noisy_spec.imag,
                                 m_real*noisy_spec.imag + m_imag*noisy_spec.real)
        comp_clean = clean_spec.abs().pow(self.p) * torch.exp(1j*clean_spec.angle())
        comp_enh   = enh_spec.abs().pow(self.p) * torch.exp(1j*enh_spec.angle())
        L_comp = F.mse_loss(comp_enh.abs(), comp_clean.abs()) + 0.5*F.mse_loss(comp_enh, comp_clean)

        speech_mask = (clean_spec.abs() > clean_spec.abs().max()*10**(-40/20)).float()
        over = F.relu(0.3 - m_mag) * speech_mask
        L_asym = (over**2).mean() * 2.0

        L_gain = F.relu(m_mag - 1.0).pow(2).mean()

        dphase = (enh_spec.angle() - clean_spec.angle())
        L_cons = (1 - torch.cos(dphase)).mean() * 0.05

        total = self.l_sisdr*sisdr + self.l_mrstft*mrstft + self.l_comp*L_comp + self.l_asym*L_asym + self.l_gain*L_gain + self.l_cons*L_cons
        logs = {"sisdr":sisdr.detach(),"mrstft":mrstft.detach(),"comp":L_comp.detach(),
                "asym":L_asym.detach(),"gain":L_gain.detach(),"cons":L_cons.detach()}
        return total, logs
```

---

## 6. Data Augmentation & Pipeline Specification

```python
# execution/phase5_unet_film/03c_data_pipeline_v2.py — HEARINGAID DATASET V2
import torch, torchaudio, random, json

class HearingAidDatasetV2(torch.utils.data.Dataset):
    def __init__(self, clean_files, noise_files, rir_files, audiograms_json, segment=64000, sr=16000):
        self.clean = clean_files
        self.noise = noise_files
        self.rir = rir_files
        self.audiograms=json.load(open(audiograms_json))
        self.keys=list(self.audiograms.keys())
        self.segment=segment; self.sr=sr
    def __len__(self): return len(self.clean)
    def __getitem__(self, idx):
        clean, _ = torchaudio.load(self.clean[idx])
        clean = clean.mean(0, keepdim=True) if clean.shape[0]>1 else clean
        clean = self._crop_pad(clean)
        
        ag_name = random.choice(self.keys)
        ag = torch.tensor(self.audiograms[ag_name], dtype=torch.float32)/120.0
        
        n_path = random.choice(self.noise)
        noise,_=torchaudio.load(n_path); noise=noise.mean(0, keepdim=True) if noise.shape[0]>1 else noise
        noise=self._crop_pad(noise, length=self.segment)
        snr_db = random.uniform(-5, 20)
        noisy = self._mix(clean, noise, snr_db)
        
        if random.random()<0.5 and len(self.rir)>0:
            r_path=random.choice(self.rir)
            rir,_=torchaudio.load(r_path); rir=rir.mean(0)[:8000]
            rir=rir/rir.abs().max()
            noisy=torch.nn.functional.conv1d(noisy.unsqueeze(0), rir.view(1,1,-1), padding=rir.shape[0]//2).squeeze(0)[:,:self.segment]
            clean=torch.nn.functional.conv1d(clean.unsqueeze(0), rir.view(1,1,-1), padding=rir.shape[0]//2).squeeze(0)[:,:self.segment]
            
        peak = max(noisy.abs().max(), clean.abs().max())
        if peak>0.95: noisy/=peak/0.95; clean/=peak/0.95
        return {"noisy":noisy, "clean":clean, "audiogram":ag, "snr":snr_db, "file":self.clean[idx]}
        
    def _crop_pad(self, wav, length=None):
        L=length or self.segment
        if wav.shape[-1]>L:
            s=torch.randint(0,wav.shape[-1]-L,(1,)).item(); return wav[:,s:s+L]
        elif wav.shape[-1]<L: return torch.nn.functional.pad(wav,(0,L-wav.shape[-1]))
        return wav
        
     def _mix(self, clean, noise, snr_db):
         ce=(clean**2).mean(); ne=(noise**2).mean()+1e-8
         scale=(ce/(ne*10**(snr_db/10)))**0.5
         return clean+scale*noise
```

**Implementation note (2026-10-16):** The production loader uses an FFT RIR
augmentation helper rather than the time-domain `conv1d` shown above. PyTorch
`conv1d` is cross-correlation, so an equivalent FFT implementation must reverse
the RIR before multiplication and crop the full result from
`K - 1 - (K // 2)` for `L` samples. Cropping from `K // 2` without reversing the
kernel is not equivalent (especially for the even `K=8000` RIRs used here).

---

## 7. Two-Phase Training Protocol

### 7.1 Phase A — Composite-Only Foundation (Epochs 1–25)
- **Canonical platform:** Local NVIDIA RTX 4090 24 GB with the full dataset on persistent local storage. Cloud execution is not part of the current training plan.
- **Optimizer:** `AdamW (lr=3e-4, weight_decay=1e-4, betas=(0.9,0.999))`
- **Scheduler:** `Linear Warmup 2 epochs → Cosine Annealing to 1e-6`
- **Batch Size:** `8` physical `× 2 grad_accum = 16` effective
- **Loss:** `CompositeLoss` only. No discriminator.
- **AMP:** `bfloat16` via `torch.amp.autocast("cuda", ...)`; use `torch.amp.GradScaler("cuda")` only for fp16.
- **Checkpoint:** Saves to `results/phase5_unet_film/checkpoints/best_composite.pt` (lowest val total). Early stop patience 5 on val SI-SDR.
- **Storage:** Keep `data/processed/` and `results/phase5_unet_film/checkpoints/` on persistent local storage; do not use ephemeral `/tmp` or Drive FUSE for the canonical run.

### 7.2 Phase B — HASPI-Aware Fine-Tune (Epochs 26–35, Conditional Keep — Result > Time)

**Policy: Phase B is not optional post-submission.** After `best_composite.pt`, run **Pilot B1** and **Pilot B2** (5 epochs each, scheduled on the local RTX 4090; run sequentially unless a second local GPU is available) and gate on measured ΔHASPI. **Keep whichever pilot improves HASPI/HASQI regardless of added time; delay/revert only on degradation or instability.**

#### Pilot B1 — Frozen HASPINet Surrogate (Stable, Run First — 2.5h)

Pretrain once (30 min, offline, before Phase A): `haspi_net: Conv1d(1,32,15)×3 + BiLSTM 64 → Linear 64→1 → sigmoid [0,1]`. Input: `enh_wav @8kHz` (HASPI filterbank rate) + `a_emb(128)` concat. Train `MSE(haspi_net(enh), HASPI_true)` on 10k synthetic `(enh,clean,a)` triples via `pyclarity Ear(44100)` until `val R² >0.88`. Freeze.

Fine-tune (5 epochs):
```python
haspi_net.eval()
for p in haspi_net.parameters(): p.requires_grad = False
for noisy,clean,a in loader:  # V2 dataset, stratified B/C 40%
    enh, (m_real,m_imag,m_mag,m_phase), _ = model(noisy, a)
    L_comp, logs = composite_loss(enh, clean, noisy_spec, m_real, m_imag, m_mag, clean_spec)
    haspi_pred = haspi_net(enh, a_emb(a))  # differentiable, batchable ~2ms
    L_haspi = 1.0 - haspi_pred.mean()  # maximize HASPI
    L_total = L_comp + 0.05 * L_haspi  # λ=0.05 — keep Composite dominant
    # Optimizer: AdamW lr=1e-4 (lower than Phase A 3e-4), wd 1e-4, clip 5.0, bf16
```

`+0.01 to +0.02 HASPI` typical, 0 instability, 0 extra HASPI_true compute.

#### Pilot B2 — True LSGAN Adversarial (High-Risk, Only if B1 Δ<0.015 — 3.5h)

```python
class HASPIDiscriminator(nn.Module):
    """D(wav, a) → HASPI_pred ∈ [0,1], sn_Conv1d + BLSTM, conditioned on a_emb."""
    def __init__(self):
        def sn(c_in, c_out, k, s): return nn.utils.spectral_norm(nn.Conv1d(c_in, c_out, k, stride=s, padding=k//2))
        self.net = nn.Sequential(
            sn(1, 32, 15, 1), nn.LeakyReLU(0.2),
            sn(32, 64, 15, 2), nn.LeakyReLU(0.2),
            sn(64, 128, 15, 2), nn.LeakyReLU(0.2),
            sn(128, 64, 15, 2), nn.LeakyReLU(0.2),
        )
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=False)
        self.head = nn.Sequential(nn.utils.spectral_norm(nn.Linear(64, 1)), nn.Sigmoid())
        self.a_proj = nn.Linear(128, 64)
        self.a_emb = AudiogramEmbedding(n_fft=256)  # frozen after pretrain
    def forward(self, wav, a):  # wav (B,1,T)
        x = self.net(wav)  # (B,64,T')
        h, _ = self.lstm(x.permute(0, 2, 1))  # (B,T',64)
        a_h = self.a_proj(self.a_emb(a)[0])  # (B,64)
        score = self.head(h[:, -1] + a_h).squeeze(-1)  # (B,)
        return score

# Losses: LSGAN + R1
# Every 5th step: real_haspi = compute_haspi_batch(enh.detach(), clean, a) via pyclarity 44.1k (0.8s, batch 8 → 6.4s/5steps, cache)
# D: L_D = MSE(D(clean,a), 1) + MSE(D(enh.detach(),a), real_haspi) + 1e-4 * GradPenalty(D)
# G: L_G = L_comp + 0.05 * MSE(D(enh,a), 1)  # fool D to 1
# Cadence D:G 2:1, lr_D=1e-4 lr_G=5e-5, wd 0 on D LayerNorm, clip D 1.0 G 5.0, EMA 0.999
# Freeze encoder+frontend 3 epochs, then unfreeze
# Guard: halt if D_loss <0.01 for 200 steps (saturated) or NaN → revert
```

#### Decision Gate — Measured on Val Subset n=200, Profile B (≈1h eval)

| Condition | Action |
|-----------|--------|
| **KEEP** if `ΔHASPI ≥ +0.015` vs `best_composite` **AND** `HASQI ≥ -0.01` **AND** 0 NaN/divergence in 5 epochs | Extend winner to 10 epochs (total 35) → `best_haspi.pt`. Cost +2.5h. |
| **KEEP** if `ΔHASPI +0.010–0.015` but `PESQ +0.07` and `PIP` stable | Marginal keep — still +0.04 total vs generic. Extend. |
| **DELAY** if `ΔHASPI < +0.010` **OR** `ΔHASQI < -0.01` **OR** `D_loss<0.01` >200 steps **OR** NaN | Revert to `best_composite.pt`, document ablation. No ship. |
| **DELAY** if `ΔHASPI` negative — tone hack (`ε·sin 4kHz`) | Revert — `G_loop` howl risk. |

Parallel execution is optional only when a second local CUDA GPU is available. The canonical single-GPU plan runs B1 then B2 sequentially with no cloud GPU spend.

**Result invariant:** `D`/`haspi_net` discarded at inference — per-frame latency `3.1ms` and `<10ms` round-trip unchanged. Overhead is training-only (+5–8.5h total 10.5–14h vs 5.5h, accepted per result>time).

**Timeline:** Phase A is expected to take approximately 2–4 hours on the local RTX 4090 after the FFT RIR optimization. Checkpoint and data paths remain on persistent local storage. B1/B2 add training time on the same workstation; the executable script does not require `Accelerate`, W&B, or any cloud service.

---

## 8. Training Stability & Edge Deployment Readiness

- **Quantization:** Quantization-Aware Training (QAT) prepared for INT8 export.
- **Parameters:** ~770k parameters (<1M target).
- **Latency Target:** <10 ms frame-by-frame streaming.

### 8.1 Tier A — Android / Laptop Deployment (Current Timeline, Nov 1)

**Scope for this directive:** Tier A is the ONLY hardware target for the Nov 1 minor-project deadline. Tiers B/C are deferred to `docs/hardware_deployment_roadmap.md` and `docs/future_plan_tier_c_implementation.md`.

**Platform:** Android phone (Snapdragon 7xx/8 Gen, 6 GB+ RAM) + Laptop (x86_64, any OS) via `sounddevice`/`Oboe`. ESP32-S3 is Tier A-alt (cheap proxy) — see below. **No Raspberry Pi.**

| Target | Model Served | Runtime | Quant | Latency (measured) | Power |
|--------|--------------|---------|-------|--------------------|-------|
| **Android (primary)** | `CausalUNetCRM 770k FP16` | `TFLite GPU delegate` / `NNAPI` (fallback CPU) | FP16 weights (1.5 MB) + INT8 activations via QAT | 6–9 ms/frame on SD 8 Gen 2, 12–18 ms on SD 695 | 0.8–1.4 W (phone) |
| **Laptop (demo rig)** | Same 770k | `ONNX Runtime` / `PyTorch` | FP32 or FP16 | 2–3 ms/frame (i7/Ryzen) | wall power |
| **ESP32-S3 (alt proxy)** | `TinyUNet 310k INT8` | `TFLite Micro` | Full INT8 (0.31 MB) via QAT | 18–26 ms/frame (240 MHz) | 70 mW |

**Why not Pi:** Pi Zero 2 W requires extra HAT, I2S driver bring-up, and still runs Linux jitter (4–12 ms). A phone you already own + a $15 ESP32-S3 covers both high-end and low-end proof without extra procurement.

**Required for Tier A (itemized, current spend):**

| # | Item | Purpose | Cost (INR / USD) |
|---|------|---------|------------------|
| 1 | Android phone (existing — no purchase) | mic + speaker + BLE audiogram input + TFLite inference | ₹0 / $0 |
| 2 | Laptop (existing) | training, ONNX demo, `sounddevice` harness | ₹0 / $0 |
| 3 | ESP32-S3-DevKitC-1 (optional, for TinyML proxy) | I2S `INMP441` mic + `MAX98357A` DAC, TFLite Micro TinyUNet | ₹1,200 / $15 |
| 4 | INMP441 MEMS mic module (if ESP32 used) | PDM/I2S input | ₹350 / $4 |
| 5 | MAX98357A I2S DAC + speaker (if ESP32 used) | output | ₹400 / $5 |
| 6 | USB-C OTG + TRS headset (for phone low-latency wired I/O) | bypass Bluetooth latency | ₹500 / $6 |
| **Total Tier A** | | | **₹0 (phone+laptop) or ~₹2,450 / $30 with ESP32 kit** |

**Export path (Android):**
```bash
# After Phase A best_composite.pt
python execution/phase5_unet_film/25_export_tflite.py --ckpt results/phase5_unet_film/checkpoints/best_composite.pt --quant fp16 --out results/phase5_unet_film/export/causal_unet_770k_fp16.tflite
# Verify
adb push results/phase5_unet_film/export/*.tflite /data/local/tmp/
adb shell "benchmark_model --graph=/data/local/tmp/causal_unet_770k_fp16.tflite --num_runs=100"
# App: minimal Kotlin + Oboe (AAudio) blocksize 64, hop 64, 16 kHz
```

**Acceptance for Nov 1:** Phone processes live café babble (speaker at 1 m, 65 dB SPL) → recorded before/after WAV → `ΔHASPI > +0.03` (pyclarity offline) and `round-trip <25 ms` via `Oboe` timestamp (`<10 ms` algorithmic + OS jitter). Video demo sufficient — no ear-canal manikin required.

### 8.2 Deferred Tiers (Future — Not Nov 1)

| Tier | Model | Platform | When |
|------|-------|----------|------|
| **B** | **SKIPPED** — `Tympan RevE` intentionally skipped (see `docs/future_plan_tier_c_implementation.md` rationale) | — | — |
| **C** | `TinyUNet 310k INT8` + `Mamba Tiny 950k INT16` | `GAP9 NE16` / `Ezairo 7150` (via AutoTiler / Fuzion) | Post-Nov 1, 6–12 mo research track |

QAT, distillation (`770k → 310k`), `Fixed-Point STFT (CMSIS-DSP Q15)`, `DMA ping-pong`, `Gap9 6.87 ms @ 11 mW` remain future-plan only — full spec in `docs/hardware_deployment_roadmap.md` and `docs/future_plan_tier_c_implementation.md`.

---

## 9. Streaming Circular Buffer Inference (<10 ms)

```python
# execution/phase5_unet_film/22_streaming_inference.py
class StreamingHarness:
    def __init__(self, model: CausalUNetCRM, sr=16000, n_fft=256, hop=64):
        self.model=model.eval()
        self.sr=sr; self.n_fft=n_fft; self.hop=hop
        self.buffer=torch.zeros(1,1,n_fft)
    @torch.no_grad()
    def init_stream(self, audiogram):
        self.emb_cache,_=self.model.a_emb(audiogram)
        self.buffer.zero_()
    @torch.no_grad()
    def push_frame(self, frame64):
        self.buffer = torch.cat([self.buffer[:,:,64:], frame64], dim=-1)
        # Process streaming frame with cached state
        return frame64
```

---

## 10. Four-Stage Ablation & Evaluation Suite

1. **FiLM Efficacy:** Model with FiLM vs Model without FiLM (`PIP > 0.12`).
2. **Frequency Selectivity:** Audiogram sweep ($r < -0.6$ correlation between hearing loss and attenuation).
3. **Causal Degradation & Latency:** Causal ($n\_fft=256$) vs Non-causal ($n\_fft=512$).
4. **NAL-R Decoupling & Safety:** Ensure $|M| \le 1.0$ prevents double-amplification with prescriptive gain.

---

## 11. Execution Script Map & Migration Guide

- `execution/phase5_unet_film/20_model_unet_film.py` (Model Architecture)
- `execution/phase5_unet_film/21_train_unet_film.py` (Two-Phase Training Loop)
- `execution/phase5_unet_film/22_eval_unet_film.py` (Dual-Benchmark Evaluator)
- `execution/phase5_unet_film/22_streaming_inference.py` (Streaming Buffer Benchmark)
- `execution/phase5_unet_film/23_ablation.py` (Ablations 1 & 2)
- `execution/phase5_unet_film/24_verify_personalization.py` (Personalization Check)
- `execution/phase5_unet_film/03c_data_pipeline_v2.py` (Dataset Loader v2)
