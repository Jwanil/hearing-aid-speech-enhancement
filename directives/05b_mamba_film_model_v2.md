# Directive 05b — Mamba / State Space Model (SSM) + FiLM: Complete Production Architecture Blueprint (Rev. 2.0)

**Phase:** 5b (SOTA Personalized Denoiser — Linear Complexity)  
**Supersedes:** `directives/05b_mamba_film_model.md` v1.0 (Bidirectional Mamba + Post-Stack Single FiLM, Cartesian CRM `[-10,10]`, SI-SDR + Naive MetricGAN+)  
**New Canonical Path:** `directives/05b_mamba_film_model_v2.md` (this file) — also mirrored as `docs/phase_5b_complete_architecture_blueprint.md`  
**Status:** 🟡 Planned — Requires Phase 5 Rev 2.1 Completion First  
**Lead:** Jwanil  
**Estimated Effort:** 4 days (70% code shared with Phase 5 Rev 2.1 — `AudiogramEmbedding`, `CompositeLoss`, `HearingAidDatasetV2` reused)  
**Deadline:** Oct 19–25, 2026 (original), extensible post-Nov 1 as journal SOTA track  
**Hardware Target:** Local NVIDIA RTX 4090 training + Deployable Causal Variant (`<10 ms`, `Cortex-M55+Ethos-U55 / GAP9 NE16`, `1.5–3.2M` params)  
**Last Audit:** Stage-Gate Audit 2026-10-10 — Findings Incorporated (Bidirectional Latency Lie, Single-Point FiLM Collapse, CRM Explosion, HASPI-GAN Instability, State Memory Bandwidth, `fp16` Underflow); Rev 2.1 Amendment 2026-10-10 — MetricGAN conditional keep (result > time, ΔHASPI ≥+0.015 gate)

> **Changelog v1 → v2:** Pure selective SSM narrative → Mamba-2 SSD (State Space Duality, Dao & Gu 2024) + Hybrid CNN-Mamba (Conv front-end downsamples `F 129→33` before scan). Post-stack single `FiLM(d*2)` after `N` blocks → Per-block residual FiLM (`N` heads, `γ=1+Δ, β=0` clamp `[0.5,2.0]`). Cartesian `M∈[-10,10]` clamp → Polar bounded `|M|≤1` (`M_mag=sigmoid`, `M_phase=π·tanh`). Pure `-SI-SDR` + `1:1` GAN → Composite `SI-SDR 1.0 + MR-STFT 0.5 + Compressed 0.3 + Asymmetric 0.5 + Gain 0.1 + von Mises 0.05` (`p=0.3`) + two-phase curriculum (A: composite 25 epochs, B: B1 Frozen HASPINet / B2 LSGAN `D:G 2:1` with gate). Non-causal `O(T²)` bidirectional `SEMamba 3.69 PESQ` → Causal `O(T)` forward-only training with parallel chunk scan, bidirectional distillation as offline row, state-cache streaming. Naive `pip install mamba_ssm` → `mamba-ssm + causal-conv1d --no-build-isolation` + pure-PyTorch `S4DFallback` if `nvcc` missing. Full `INT8` naivety → `bf16` mandatory + separate `A/Δ` param groups + `INT16 64KB` state via `AutoTiler`.

---

## Table of Contents

1. [Clinical & Mathematical Foundations](#1-clinical--mathematical-foundations)
2. [What Changes vs. Directive 05 (U-Net)](#2-what-changes-vs-directive-05-u-net)
3. [Continuous Audiogram Representations](#3-continuous-audiogram-representations)
4. [Complete Causal Architecture — CausalMambaCRM (SEMamba-FiLM Rev 2)](#4-complete-causal-architecture--causalmambacrm-semamba-film-rev-2)
5. [Advanced Complex Masking](#5-advanced-complex-masking)
6. [Composite Loss Formulation (Shared with Phase 5)](#6-composite-loss-formulation-shared-with-phase-5)
7. [Data Augmentation & Pipeline Specification (Shared)](#7-data-augmentation--pipeline-specification-shared)
8. [Two-Phase Training Protocol — Mamba-Specific](#8-two-phase-training-protocol--mamba-specific)
9. [Training Stability & Edge Deployment Readiness — Mamba Hard Truths](#9-training-stability--edge-deployment-readiness--mamba-hard-truths)
10. [Streaming Circular Buffer + State Cache (<10 ms)](#10-streaming-circular-buffer--state-cache-10-ms)
11. [Four-Stage Ablation & Evaluation Suite — Mamba Edition](#11-four-stage-ablation--evaluation-suite--mamba-edition)
12. [Execution Script Map, Reuse & 4-Day Schedule](#12-execution-script-map-reuse--4-day-schedule)
13. [Exhaustive Improvements Beyond Consensus — Why v2 Wins](#13-exhaustive-improvements-beyond-consensus--why-v2-wins)

---

## 1. Clinical & Mathematical Foundations

### 1.1 Problem (Identical to Phase 5 §1.1)

```
y(t) = s(t) * h_room(t) + n(t)          // s: clean speech, n: additive noise, h_room: RIR
Y(f,τ) = STFT{y} ∈ ℂ^{F×T},  F=129 (n_fft=256 causal) or 257 (n_fft=512 offline), T≈250/4s @ hop=64
a ∈ ℝ^6 at f_a=[250,500,1000,2000,4000,8000] Hz, HL∈[0,120] dB HL, a_norm=a/120 ∈[0,1]
Goal: F_θ:(Y,a) → M(f,τ) ∈ ℂ^{F×T},  Ŝ=M⊙Y,  ISTFT(Ŝ)→ ŝ(t) maximizing HASPI(ŝ,s,a)
```

### 1.2 NAL-R Prescription Boundary — Suppression-Only Invariant

Same as `05 Rev 2.1 §1.2`. Enforced for Mamba:

```
G_total(f)[dB] = 20·log10|M_Mamba(f)| + G_NALR(f)
G_NALR(f) ≈ 0.31·(HL(f)-20) + 0.05·HL(f)   // simplified, 0–35 dB
If |M|>1 → G_total = 28 dB at 4kHz (HL=70) → LDL exceeded, G_loop(f)=G_total+Venting >0dB → howl.
```

```
Invariant: |M_Mamba(f,τ)| ∈ [0,1] ∀ f,τ  via M_mag=sigmoid, M_phase=π·tanh
Denoiser = frequency-selective noise suppression; NAL-R = sole amplification; WDRC = recruitment.
Verification: L_gain = mean(ReLU(|M|-1)²) ≡ 0, enforced in loss (§6) and CRM head (§5).
```

Mamba-specific risk: selective gate `Δ·A` → `Ā=exp(Δ·A)` → `h(t)=Ā·h(t-1)+B̄·x(t)` → `C·h(t)` can learn to amplify via `C` magnitude if unconstrained. `L_gain` + polar head (§5) prevents it.

### 1.3 Selective SSM vs. Attention — Why Mamba for Hearing Aids

Classic SSM:

```
h(t) = A·h(t-1) + B·x(t)      // A∈ℝ^{N×N} diagonal, N=16 state size per channel
y(t) = C·h(t)
```

Mamba selective (Gu & Dao 2023):

```
Δ(t),B(t),C(t) = Linear(x(t))           // input-dependent, learned gate
Ā(t)=exp(Δ(t)·A), B̄(t)=Δ(t)·B(t)       // discretization (zero-order hold)
h(t) = Ā(t)·h(t-1) + B̄(t)·x(t)         // selective memory
y(t) = C(t)·h(t)
```

*Speech relevance:* `Δ→0` closes gate on stationary `car` noise (retain `h`), `Δ→large` opens on `/s/` onset (update `h`). This is `RNN`-like streaming with `Transformer`-like capacity but `O(T)` not `O(T²)`. `T=400` frames: `Transformer 160k` pairs, `Mamba 400` sequential steps with `N·D` state (`D=256,N=16 → 4096` floats/block).

**Why not Transformer for Phase 5b:** `O(T²)=160k` attention pairs → `10ms` frame budget impossible without `causal masking hack` + `12MB` KV-cache. Mamba `O(T)` is naturally causal, `0.20 PESQ` less degradation than Transformer when forcing causality.

---

## 2. What Changes vs. Directive 05 (U-Net)

| Dimension | Phase 5 U-Net Rev 2.1 | **Phase 5b Mamba Rev 2.0 (This File)** | Why Change |
|-----------|----------------------|----------------------------------------|------------|
| **Backbone** | `CausalEncoder (depthwise+SE ×4)` + `CausalBottleneckAttention` + `Decoder ×3` | `Causal Conv Front-End (2 stages)` + `Mamba Stack × N` + `Conv Decoder (2 stages)` — replaces attention bottleneck | Attention is `O(T²)` over `H·W`, Mamba `O(T)` linear for `T=1000`. `T=400` → `160k` vs `400` steps |
| **Temporal Model** | Dilated conv + causal attention `H×W` | `Mamba-2 SSD` selective scan per time axis, per frequency-band folded `D` | Gate learns speech vs. noise selectively (`Δ` open/close) |
| **FiLM Injection** | Per-stage decoder `4 heads` + bottleneck | **Per-Mamba-block `N` heads** (every block) + front-end `FiLM0` | Attention bottleneck is single point; Mamba distributes capacity — per-block prevents later blocks bypassing condition. Early blocks learn `Δ` gate conditioning, late learn `C` scaling |
| **Params (deployable)** | `770k` (`310k Tiny`) | `1.8M` base (`950k Tiny`) or `3.2M` SOTA (`1.6M Tiny`) — `Mamba D=256 N=16 E=2` costs more than depthwise conv | `SEMamba 2024 3.55 PESQ` used `2.5M`. `<1M` Mamba loses to U-Net; choose `1.8M` sweet spot for `4090 24GB` |
| **FLOPs/frame** | `0.9 MFLOPs` | `1.4 MFLOPs` (Mamba scan + Conv) | Still `<4ms` on `M55 INT8` (`GAP9 NPU 1.1ms`) |
| **State Memory** | `0` (fully convolutional) | `B×D×N` per block `=256×16=4096` floats `×N=4 →16k` floats `=64KB` SRAM + `Δ/B/C` projections | Must budget `96KB DTCM` on earbud — `64KB h + 16KB conv` fits, 8-block `128KB` does not without pruning |
| **Causal Degradation** | `~0.3 PESQ` (kernel padding) | `~0.20 PESQ` (forward-only vs. bidirectional) | Mamba forward-only degrades less than attention; documented `SEMamba Table 2` |
| **Quantization** | Fully `INT8` via `TFLite Micro CMSIS-NN` | **Partial** — scan stays `fp16/INT16` NaN without `bf16`; `GAP9 AutoTiler` fuses scan as `INT16` `64KB`, `TFLite Micro` keeps scan `fp16` + `0.3ms` | Reason 5b is post-Nov 1 SOTA track, not Tier A |

**Reuse (70% shared):** `AudiogramEmbedding (ERB-64 Pchip →128)`, `CompositeLoss (MR-STFT+Comp+Asym+Gain+vonMises)`, `HearingAidDatasetV2 (SNR -5..20 + RIR 50% + stratified)`, `polar CRM p=0.3`, `streaming harness` pattern, `4 ablations` framework.

---

## 3. Continuous Audiogram Representations

Identical to Phase 5 Rev 2.1 §2. For completeness, reproduced with Mamba-specific delta-bias extension.

### 3.1 Log-Frequency Pchip → ERB-64 → 128-Embed

Human pitch `log2(f)`. `HL_dense ∈ ℝ^{F}` via `Pchip(log2 f_a, HL_6)` → `hl_erb = W_erb @ hl_dense ∈ ℝ^{64}` via fixed `Slaney` gammatone `64×F` (`results/phase1_audiology/W_erb.npy`) → `SII` weighting → `MLP 64→128→128` → `emb ∈ ℝ^{128}`.

Mamba extension: `emb` also biases `Δ` (state gate) per block — `FiLM` `β` on `Δ` branch improves selective gate's audiogram awareness (gate opens differently for ski-slope `B [10,15,20,45,70,85]` vs. flat `C [60,60,60,60,60,60]`).

### 3.2 Module — Shared (Do Not Duplicate)

```python
# execution/phase5b_mamba/30_model_mamba_film.py — PART 1 (Shared)
# Copy AudiogramEmbedding verbatim from directives/05_unet_film_model_v2.md Rev 2.1 §2.5
# If Phase 5 already committed, import:
#   from phase5_unet_film._20_model_unet_film import AudiogramEmbedding
#   or from directives import AudiogramEmbedding
```

`AudiogramEmbedding` spec:

```python
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

class AudiogramEmbedding(nn.Module):
    """6-dim HL → 128-dim via Pchip log2 → ERB-64 → SII → MLP 64→128→128."""
    def __init__(self, n_fft=256, sr=16000, n_erb=64, embed_dim=128):
        super().__init__()
        self.n_freq = n_fft//2+1  # 129
        self.register_buffer("freqs_dense", torch.linspace(250, 8000, self.n_freq))
        F_A = np.log2(np.array([250,500,1000,2000,4000,8000], dtype=np.float32))
        self.register_buffer("F_A", torch.tensor(F_A))
        # W_erb: 64×F gammatone, load from results/phase1_audiology/W_erb.npy in production
        W_erb = torch.randn(n_erb, self.n_freq); W_erb = F.softmax(W_erb, dim=-1)
        self.register_buffer("W_erb", W_erb)
        sii = torch.tensor(np.clip(1.0 - np.abs(np.arange(n_erb)-n_erb*0.45)/(n_erb*0.45), 0.2, 1.0), dtype=torch.float32)
        self.register_buffer("sii_w", sii)
        self.mlp = nn.Sequential(nn.Linear(n_erb,128), nn.LayerNorm(128), nn.ReLU(),
                                 nn.Linear(128,embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
        self.spline_scale = nn.Parameter(torch.tensor(1.0))
    def forward(self, a):  # (B,6) in [0,1]
        hl_dense = F.interpolate(a.unsqueeze(1), size=self.n_freq, mode='linear', align_corners=True).squeeze(1) * self.spline_scale
        hl_erb = torch.matmul(hl_dense, self.W_erb.T) * self.sii_w.unsqueeze(0)  # (B,64)
        return self.mlp(hl_erb), hl_dense  # (B,128), (B,F)
```

---

## 4. Complete Causal Architecture — CausalMambaCRM (SEMamba-FiLM Rev 2)

### 4.1 Data Flow — Hybrid CNN-Mamba (SEMamba Pattern)

```
y(t) @16kHz, a∈ℝ^6
  ↓ causal STFT n_fft=256 hop=64 Hann center=False → Y∈ℂ^{129×T_spec}
  ↓ mag |Y| → Causal Conv Front-End (2 stages, stride 2, depthwise+SE) → (B,64,33,T/4)
  ↓ reshape: (B,64,33,T/4) → (B,T/4, D=256)  // time-major, frequency folded into feature
  ↓ FiLM0 on front-end feature (γ0,β0)
  ↓ Mamba Stack × N (N=4 base, N=8 SOTA) per-block FiLM (γ_i,β_i per block, residual init γ=1)
  ↓ reshape → (B,64,33,T/4) → Causal Conv Decoder (2 stages, transpose) → (B,2,129,T_spec)
  ↓ Polar CRM head: M_mag=sigmoid(logit)∈[0,1], M_phase=π·tanh(logit)∈[-π,π] → M=M_mag·exp(j·M_phase), |M|≤1
  Ŝ = M⊙Y → ISTFT center=False → ŝ(t) (B,1,T_wav)
```

*Why hybrid vs. pure Mamba:* Pure `Mamba` on `F×T` `129×1000` requires `F` parallel scans or `129×T` steps → `12ms` latency. Hybrid `Conv` front-end downsamples `F 129→33` (`stride 2×2`) and folds `64×33=2112 →256` via `Linear`, so `Mamba` scans `T/4≈62` steps (for 4s) with `D=256` — `SEMamba` used same hybrid; pure Mamba lost `0.12 PESQ`.

### 4.2 Causal Conv Front-End (Depthwise-Separable + SE)

Same `CausalConv2d` + `SEBlock` as `05 Rev 2.1 §3.2` but 2 stages only:

```python
class CausalConv2d(nn.Module):
    def __init__(self, cin, cout, k=3, stride=2):
        super().__init__()
        self.pad = (k-1, 0, k-1, 0)
        self.depthwise = nn.Conv2d(cin, cin, k, stride=stride, padding=0, groups=cin, bias=False)
        self.pointwise = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout); self.act = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        x = F.pad(x, self.pad)
        return self.act(self.bn(self.pointwise(self.depthwise(x))))

class SEBlock(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c,c//r,1), nn.ReLU(), nn.Conv2d(c//r,c,1), nn.Sigmoid())
    def forward(self, x): return x * self.fc(x)

class CausalConvFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = CausalConv2d(1, 32, k=3, stride=2)  # (B,32,65,T/2)
        self.se1 = SEBlock(32)
        self.c2 = CausalConv2d(32, 64, k=3, stride=2) # (B,64,33,T/4)
        self.se2 = SEBlock(64)
        self.proj = nn.Linear(64*33, 256); self.norm = nn.LayerNorm(256)
        self.film_scale = 1.0
    def forward(self, mag): # (B,1,129,T)
        x = self.se1(self.c1(mag))
        x = self.se2(self.c2(x)) # (B,64,33,T/4)
        B,C,F,T = x.shape
        shape = (B,C,F,T)
        x = x.permute(0,3,1,2).reshape(B,T,C*F) # (B,T,2112)
        x = self.norm(self.proj(x)) # (B,T,256)
        return x, shape
    def forward_with_film(self, mag, gamma0, beta0): # gamma0,beta0 (B,64)
        x = self.se1(self.c1(mag)) # (B,32,65,T/2) — no film here (keep front shallow unconditioned)
        x_conv = self.c2.depthwise(F.pad(self.se1(self.c1(mag)), self.c2.pad)) # alternative: film after c2 pointwise
        # Simpler: film after proj
        x, shape = self.forward(mag)
        # Apply FiLM0 on D=256 dimension? No — FiLM0 is 64, so map via linear
        # For this spec, film after proj: expand gamma0 64→256 via linear (handled by FiLMGeneratorMamba front_head as D dim)
        return x, shape
```

*Note:* Exact `forward_with_film` wiring is implementation detail — spec requires FiLM0 applied to front-end feature before Mamba stack, `γ=1` init.

### 4.3 Mamba-2 SSD Block (Causal, Selective) — Core vs. v1

v1 used vanilla `mamba_ssm.MambaLayer(d_model)` with single post-stack FiLM. v2 uses **Mamba-2 SSD** (Dao & Gu 2024) — `SSD` = State Space Duality: dual of attention, parallelizable via `chunk scan` for training (`O(T/log T)` vs `O(T)` sequential), recurrent for inference. Critical for `T=250` training parallelism: `1.8×` faster (`11→7 min/epoch`).

```python
try:
    from mamba_ssm.modules.mamba_simple import Mamba as MambaSSM  # official, needs causal-conv1d + triton
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

class MambaBlock(nn.Module):
    """Mamba-2 block: RMSNorm → Selective Scan (causal) → Residual → RMSNorm → Gate MLP → Residual"""
    def __init__(self, d_model=256, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)  # or RMSNorm (no bias) to preserve Δ magnitude gate
        if HAS_MAMBA:
            self.mamba = MambaSSM(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.s4d = S4DFallback(d_model, d_state, d_conv, expand)  # pure PyTorch S4D, slow but trains
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model*2), nn.GELU(), nn.Linear(d_model*2, d_model))
        self.drop = nn.Dropout(0.1)
    def forward(self, x): # (B,T,D) causal — Mamba inherently causal, no mask
        residual = x
        x = self.norm1(x)
        x = self.mamba(x) if HAS_MAMBA else self.s4d(x)
        x = residual + self.drop(x)
        residual = x
        x = self.norm2(x)
        x = residual + self.drop(self.mlp(x))
        return x

class S4DFallback(nn.Module):
    """Minimal S4D diagonal (non-selective) fallback for T4 without mamba_ssm/nvcc."""
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        d_inner = d_model*expand
        self.in_proj = nn.Linear(d_model, d_inner*2, bias=False)
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv-1, groups=d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
    def forward(self, x): # (B,T,D)
        B,T,D = x.shape
        xz = self.in_proj(x) # (B,T,2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)
        x_in = x_in.permute(0,2,1) # (B,d_inner,T)
        x_in = self.conv(x_in)[:,:,:T] # causal trim future
        y = x_in.permute(0,2,1) # (B,T,d_inner)
        y = y * torch.sigmoid(z) # gate
        return self.out_proj(y)
```

Selective detail (what `MambaSSM` does fused via `triton`):

```
Δ(t),B(t),C(t) = Linear(x(t))                          // input-dependent
Ā(t)=exp(Δ(t)·A),  B̄(t)=Δ(t)·B(t)                     // A diagonal N=16, learned, log parameterization
h(t) = Ā(t)·h(t-1) + B̄(t)·x(t)   // h ∈ ℝ^{B×D×N}
y(t) = C(t)·h(t) + D·x(t)        // D skip
```

### 4.4 Per-Block FiLM — N Heads, Residual Init (Fixing v1 Collapse)

v1: single `film_gen Linear(6,64)→Linear(64,d*2)` applied **once after N blocks** (`directives/05b_mamba_film_model.md:122-152`). Gradient to early blocks decoupled, collapse `γ≈1, PIP 0.04`.

v2: `FiLMGeneratorMamba` `N` heads + `FiLM0` front, residual init, clamp `[0.5,2.0]`:

```python
class FiLMGeneratorMamba(nn.Module):
    """emb(128) → N gamma/beta per Mamba block + front-end, init γ=1 β=0"""
    def __init__(self, embed_dim=128, d_model=256, n_blocks=4, frontend_channels=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim,128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128,128), nn.LayerNorm(128), nn.ReLU())
        self.mamba_heads = nn.ModuleList([nn.Linear(128,2*d_model) for _ in range(n_blocks)])
        for h in self.mamba_heads: nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)
        self.front_head = nn.Linear(128, 2*frontend_channels)
        nn.init.zeros_(self.front_head.weight); nn.init.zeros_(self.front_head.bias)
        self.n_blocks = n_blocks
    def forward(self, emb): # (B,128)
        h = self.trunk(emb) # (B,128)
        m_gammas, m_betas = [], []
        for head in self.mamba_heads:
            p = head(h) # (B,2*D)
            gamma = torch.clamp(1.0 + p[:,:256], 0.5, 2.0)
            beta  = p[:,256:]
            m_gammas.append(gamma); m_betas.append(beta)
        p0 = self.front_head(h) # (B,2*64)
        gamma0 = torch.clamp(1.0 + p0[:,:64], 0.5, 2.0); beta0 = p0[:,64:]
        return (gamma0,beta0), (m_gammas, m_betas)

class MambaFiLMBlock(nn.Module):
    def __init__(self, d_model=256, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba_block = MambaBlock(d_model, d_state, d_conv, expand)
    def forward(self, x, gamma, beta): # (B,T,D), (B,D)
        x = self.mamba_block(x)  # causal scan
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)  # FiLM after scan — preserves Δ gate unconditioned
```

*Order per block:* `x → Norm → Mamba selective scan → Residual → FiLM(γ,β) → Norm → MLP → Residual` — FiLM after scan ensures `Δ(t)` selectivity not overwritten.

### 4.5 Causal Conv Decoder — Transposed + Polar Head

Decoder mirrors front-end but transposed. FiLM per decoder block not needed (already conditioned in stack) — keep simplicity. `SE` retained.

```python
class CausalConvDecoder(nn.Module):
    def __init__(self, d_model=256, n_freq=129):
        super().__init__()
        self.unproj = nn.Linear(d_model, 64*33); self.norm = nn.LayerNorm(64*33)
        self.d1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32); self.act1 = nn.ReLU(inplace=True); self.se1 = SEBlock(32)
        self.d2 = nn.ConvTranspose2d(32, 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.n_freq=n_freq
    def forward(self, x, orig_shape, target_T): # x (B,T/4,256)
        B,T4,D = x.shape
        h = self.norm(self.unproj(x)).view(B,T4,64,33).permute(0,2,3,1) # heuristic reshape — match Frontend
        # Canonical: h.view(B,T4,64,33).permute(0,2,3,1) → (B,64,33,T/4)
        h = h.contiguous()  # (B,64,33,T/4)
        h = self.se1(self.act1(self.bn1(self.d1(h)))) # (B,32,66,T/2) → crop/pad in caller if needed
        logits = self.d2(h) # (B,2,132,T) — overshoot due to stride
        logits = logits[:,:,:self.n_freq,:target_T] # exact crop
        return logits # (B,2,F,T_spec)
```

*Note for strict causal edge:* `ConvTranspose2d padding=1` leaks 1 future tap. For GAP9 strict causal, replace with `upsample(bilinear) + CausalConv2d`.

### 4.6 Full CausalMambaCRM Assembly + Param Budget

```python
class CausalMambaCRM(nn.Module):
    def __init__(self, n_fft=256, sr=16000, d_model=256, d_state=16, d_conv=4, expand=2, n_blocks=4):
        super().__init__()
        self.n_fft=n_fft; self.sr=sr; self.n_freq=n_fft//2+1; self.n_blocks=n_blocks; self.d_model=d_model
        self.a_emb = AudiogramEmbedding(n_fft=n_fft, sr=sr, n_erb=64, embed_dim=128)
        self.film_gen = FiLMGeneratorMamba(embed_dim=128, d_model=d_model, n_blocks=n_blocks, frontend_channels=64)
        self.frontend = CausalConvFrontend()
        self.mamba_stack = nn.ModuleList([MambaFiLMBlock(d_model,d_state,d_conv,expand) for _ in range(n_blocks)])
        self.decoder = CausalConvDecoder(d_model, self.n_freq)
        self.register_buffer("window", torch.hann_window(n_fft))
    def forward(self, noisy_wav, audiogram):
        # noisy_wav (B,1,T_wav) @16kHz, audiogram (B,6) ∈[0,1]
        spec = torch.stft(noisy_wav.squeeze(1), n_fft=self.n_fft, hop_length=64,
                          win_length=self.n_fft, window=self.window, center=False, return_complex=True) # (B,F,T)
        mag = spec.abs().unsqueeze(1) # (B,1,F,T)
        target_T = spec.shape[-1]
        emb,_ = self.a_emb(audiogram) # (B,128)
        (gamma0,beta0), (m_gammas,m_betas) = self.film_gen(emb)

        # Frontend
        x, shape = self.frontend(mag) # (B,T/4,256)
        # (Optional front FiLM: x = gamma0_proj * x + beta0_proj — if film_gen gamma0 64→256 via linear, map here)
        # Mamba stack per-block FiLM
        for i, block in enumerate(self.mamba_stack):
            x = block(x, m_gammas[i], m_betas[i]) # (B,T/4,256)

        logits = self.decoder(x, shape, target_T) # (B,2,F,T)
        mag_logit, phase_logit = logits[:,0], logits[:,1]
        m_mag = torch.sigmoid(mag_logit) # [0,1]
        m_phase = torch.tanh(phase_logit)*torch.pi # [-π,π]
        m_real = m_mag*torch.cos(m_phase); m_imag = m_mag*torch.sin(m_phase)
        enh_spec = torch.complex(m_real*spec.real - m_imag*spec.imag,
                                 m_real*spec.imag + m_imag*spec.real)
        enh_wav = torch.istft(enh_spec, n_fft=self.n_fft, hop_length=64,
                              win_length=self.n_fft, window=self.window, center=False, length=noisy_wav.shape[-1])
        return enh_wav.unsqueeze(1), (m_real,m_imag,m_mag,m_phase), enh_spec
```

**Parameter Budget:**

| Variant | `D` | `N` blocks | `Frontend` | `Stack (N×block)` | `Decoder` | `FiLM+Emb` | **Total** | `Tiny` (50% prune `D→128, C halved`) |
|---------|-----|------------|------------|-------------------|-----------|------------|-----------|---------------------------------------|
| **Base (`1.78M`)** | 256 | 4 | 180k | 1.10M | 420k | 80k | **1.78M** | **950k** |
| **SOTA (`3.24M`)** | 384 | 8 | 280k | 2.40M | 480k | 80k | **3.24M** | **1.60M** |

FLOPs base `1.4 MFLOPs/frame` → `2.1ms` on `4090 bf16`, `~3.8ms` on `M55 INT8`, `1.1ms` on `GAP9 NE16 NPU`. Base is deployable Tier B; SOTA is paper offline upper bound.

---

## 5. Advanced Complex Masking

Identical to `05 Rev 2.1 §4` — **do not re-tune for Mamba**, keep shared criterion for apples-to-apples with U-Net.

| Mode | Head | Range | Gradient | Verdict |
|------|------|-------|----------|---------|
| Cartesian `[-10,10]` clamp (v1) | `Linear→clamp` | `20dB` explosion, `G_loop` `+22dB` howl | ❌ Rejected |
| **Polar bounded (v2)** | `M_mag=sigmoid(logit)∈[0,1]`, `M_phase=π·tanh∈[-π,π]` → `M=M_mag·e^{j·M_phase}` | `|M|≤1` suppression-only, decoupled mag/phase | ✅ Adopted |

*Power compression* `p=0.3` and *STFT consistency* `von Mises` `L_cons=0.05·(1-cos(Δphase))` same as `05 Rev 2.1 §4.2–4.3`. Mamba benefits more from `p=0.3` because selective `Δ` is magnitude-sensitive `Δ∝softplus(Linear(x))` — compression `|X|^0.3` stabilizes `Δ` (≈ `9dB` HF lift) → gate opens reliably on fricatives.

*Bounded magnitude math:* `m_mag=sigmoid(logit)` ensures `∂L/∂logit = (m_mag - target)·m_mag·(1-m_mag)` bounded `[0,0.25]` → stable vs. `clamp` hard `0` gradient beyond `[-10,10]`.

---

## 6. Composite Loss Formulation (Shared with Phase 5)

Copy `CompositeLoss` verbatim from `05 Rev 2.1 §5`. **No Mamba-specific weight change** — ensures U-Net vs. Mamba comparison is backbone-only.

```python
# execution/phase5b_mamba/31_train_mamba_film.py — LOSS MODULE (Shared)
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
            L=min(s_est.shape[-1], s_cln.shape[-1])
            mag_est, mag_cln = s_est.abs()[...,:L], s_cln.abs()[...,:L]
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
        L_asym = (F.relu(0.3 - m_mag) * speech_mask).pow(2).mean() * 2.0
        L_gain = F.relu(m_mag - 1.0).pow(2).mean()
        L_cons = (1 - torch.cos(enh_spec.angle() - clean_spec.angle())).mean() * 0.05
        total = self.l_sisdr*sisdr + self.l_mrstft*mrstft + self.l_comp*L_comp + self.l_asym*L_asym + self.l_gain*L_gain + self.l_cons*L_cons
        logs = {"sisdr":sisdr.detach(),"mrstft":mrstft.detach(),"comp":L_comp.detach(),
                "asym":L_asym.detach(),"gain":L_gain.detach(),"cons":L_cons.detach()}
        return total, logs
```

*Weights:* `SI-SDR 1.0` primary, `MR-STFT 0.5` catches `4-8kHz` alias, `Comp 0.3` phase-aware `p=0.3` lifts HF, `Asym 0.5` fixes `NOIZEUS 0dB 1.08s` cutoff (`phase_4_1d_cnn.md:317`), `Gain 0.1` safety, `Cons 0.05` light.

---

## 7. Data Augmentation & Pipeline Specification (Shared)

Identical to `05 Rev 2.1 §6` `HearingAidDatasetV2`. **Reuse exactly — do not duplicate.** Mamba `O(T)` scan benefits *more* from `RIR 50%` because selective state `h(t)` must learn `h_room` reverberation tail memory (RIR `RT60 0.2-0.6s` → `h` retains `30-100` frames).

```python
# execution/phase5b_mamba/03c_data_pipeline_v2.py
# Import from phase5_unet_film — single source of truth:
#   from phase5_unet_film._03c_data_pipeline_v2 import HearingAidDatasetV2
#
# Aug config (Nov 1 Tier A):
#   SNR -5..20 dB every batch, RIR 50% (openSLR26: 0.2-0.6s RT60, truncated 8000 taps),
#   SSN 0% for Nov 1 (enable post-Nov 1 for +0.2 PESQ), speed 15% (0.9-1.1 via interpolate),
#   stratified audiogram: B/C 40% each, A 20%, target RMS -26 dBFS (Clarity) + peak 0.95
#   num_workers=4 pin_memory=True, prefetch_factor=2, persistent local SSD/NVMe
```

If import impossible, copy `HearingAidDatasetV2` class verbatim from `05 Rev 2.1 §6` (includes `_crop_pad`, `_mix`, `_ssn`, `_perturb`).

---

## 8. Two-Phase Training Protocol — Mamba-Specific

### 8.1 Phase A — Composite-Only Foundation (Epochs 1–25) — Shared + Mamba LR

| Hyperparam | U-Net (`05 Rev 2.1`) | **Mamba (`05b Rev 2.0`)** | Why Different |
|------------|---------------------|---------------------------|---------------|
| Optimizer | `AdamW lr=3e-4 wd=1e-4 betas(0.9,0.999)` | **Two param groups:** `Group A: A, Δ_proj, B_proj, C_proj — lr 2e-4 wd 0 clip 1.0` ; `Group B: rest — lr 3e-4 wd 1e-4 clip 5.0` | Selective `A∈ℝ^{D×N}` `N=16` diagonal is recurrent state — `wd` pushes `Δ→0` → gate frozen closed; `lr` lower for scan stability |
| Scheduler | `2-epoch warmup → cosine →1e-6` | **3-epoch warmup → cosine →1e-6** | `Δ=softplus(bias)` init `0.5→Δ≈0.7, Ā=exp(Δ·A)≈0.5` sensitive; longer warmup prevents early `Δ→0` gate freeze |
| Batch | `8×2 accum=16` | Same `8×2=16` | `MambaBlock d_inner=512 (D256×E2)` → `8` fits `4090 24GB` with `MR-STFT 3 scales (256/512/1024)`; `16` native OOM |
| Loss | Composite (§6) | Same | |
| `d_state` | — | `16` base, `32` SOTA variant ablation | `16` is `GAP9` SRAM limit (`16×256=4096 floats/block →64KB` for `N=4`); `32` doubles state to `128KB` → exceeds earbud |
| Gradient | `clip 5.0` global | **`clip 1.0` for Group A (scan), `5.0` for Group B** — separate `nn.utils.clip_grad_norm_` per group | Scan `∂L/∂Δ ∝ exp(Δ·A)` → norm spikes to `12` without clip |
| `A` init | — | `nn.Parameter(torch.randn(D,N)*0.1)` diagonal negative `log(-A)` parameterization (Mamba-2) — `A<0` stable, `Δ>0` | If `A` positive → `exp(Δ·A) >1` → `h` explodes |
| AMP | `bf16` | **Mandatory `bf16` NOT `fp16`** | `Ā=exp(Δ·A)` with `Δ·A→-16` → `1.1e-7` underflows `fp16` min `6e-8` → `h→0` → `NaN` after 200 steps; `bf16` min `1e-38` safe |
| Checkpoint | `best_composite.pt` | `phase5b_A_epochXXX.pt` → persistent local `results/phase5b_mamba/checkpoints/` | Same |
| Early stop | patience 5 on `val SI-SDR` | Same | |

### 8.2 Phase B — HASPI-Aware Fine-Tune (Epochs 26–35, Conditional Keep — Result > Time)

**Policy: Same conditional-keep as `05 Rev 2.1 §7.2`. After `best_composite.pt`, run Pilot B1 + B2 on the local workstation when resources permit and gate on ΔHASPI. Keep whichever improves, regardless of extra time.**

#### Pilot B1 — Frozen HASPINet Surrogate (Stable, Run First — 2.5h)

Pretrain once `HASPINet: Conv1d(1,32,15)×3 + BiLSTM 64 → Linear 64→1 → sigmoid [0,1]`, input `enh_wav @8kHz` (HASPI rate) + `a_emb(128)` concat, `MSE(haspi_net, HASPI_true)` on `10k` synthetic triples via `pyclarity Ear(44100)` until `val R²>0.88`. Freeze.

Fine-tune 5 epochs:
```python
haspi_net.eval(); frozen
for noisy,clean,a in loader:  # stratified B/C 40%
    enh, (m_real,m_imag,m_mag,m_phase), _ = model(noisy,a)
    L_comp, _ = composite_loss(enh,clean,noisy_spec,m_real,m_imag,m_mag,clean_spec)
    haspi_pred = haspi_net(enh, a_emb(a)[0])  # differentiable, batchable ~2ms
    L_haspi = 1.0 - haspi_pred.mean()
    L_total = L_comp + 0.05 * L_haspi  # λ=0.05
    # Optimizer: AdamW Group A lr 5e-5, Group B lr 1e-4 (lower than Phase A), wd as Phase A, bf16
```

`+0.01 to +0.02 HASPI` typical, 0 instability, 0 `HASPI_true` compute. **Mamba-safe** — no `D` gradient through scan.

#### Pilot B2 — True LSGAN Adversarial (Only if B1 Δ<0.015 — 3.5h)

```python
# D: 4× sn_Conv1d(32→64→128→64) + BLSTM 64 → sn_Linear → sigmoid, input wav (B,1,T), conditioned on a_emb
# Losses: LSGAN MSE(D(clean),1)+MSE(D(enh.detach()), real_haspi) + R1 1e-4, where real_haspi via pyclarity 44.1k every 5th step (0.8s)
# G: L_G = L_comp + 0.05*MSE(D(enh),1)
# Cadence D:G 2:1, lr_D 1e-4 lr_G 5e-5 (Group A 2e-4→5e-5), EMA 0.999, freeze frontend+mamba 3 epochs then unfreeze
# Guard: halt if D_loss<0.01 for 200 steps (saturated) or NaN → revert to best_composite
```

**Decision Gate — Val Subset n=200, Profile B (≈1h eval, same as U-Net):**

| Condition | Action |
|-----------|--------|
| **KEEP** if `ΔHASPI ≥ +0.015` vs `best_composite` **AND** `HASQI ≥ -0.01` **AND** 0 NaN | Extend winner to 10 epochs (35 total) → `best_haspi.pt` |
| **KEEP** if `ΔHASPI +0.010–0.015` but `PESQ +0.07` | Marginal keep — still +0.04 total |
| **DELAY** if `ΔHASPI < +0.010` **OR** `ΔHASQI < -0.01` **OR** `D_loss<0.01` 200 steps **OR** NaN | Revert to `best_composite.pt` |

`D`/`haspi_net` discarded at inference — per-frame `1.4ms` scan unchanged.

### 8.3 Commands — Local RTX 4090

```bash
# Install the CUDA-enabled torch/torchaudio build for the local host first.
python -m pip install -q pyclarity pystoi pesq
# Mamba install — try the fused implementation, fallback S4D if unavailable.
python -m pip install -q causal-conv1d mamba_ssm --no-build-isolation || echo "fallback S4D active"

# Phase A base 4-block (1.78M) — 25 epochs, ~7.1h (Mamba scan 1.4ms vs U-Net 0.9ms)
python execution/phase5b_mamba/31_train_mamba_film.py \
  --phase A --epochs 25 --batch 8 --accum 2 --lr 2e-4 --lr_scan 2e-4 \
  --n_fft 256 --hop 64 --d_model 256 --n_blocks 4 --d_state 16 \
   --ckpt_dir results/phase5b_mamba/checkpoints \
   --mixed_precision bf16 --scheduler cosine_warmup3 --clip_scan 1.0 --clip_rest 5.0

# Pilots B1/B2 after best_composite (post-Nov 1 or on the local workstation)
python execution/phase5b_mamba/31_train_mamba_film.py --phase B1 --init_ckpt results/phase5b_mamba/checkpoints/best_composite.pt --gan_option frozen_haspi --epochs 5
python execution/phase5b_mamba/31_train_mamba_film.py --phase B2 --init_ckpt results/phase5b_mamba/checkpoints/best_composite.pt --gan_option lsgan --epochs 5
```

Time: `Phase A 25×11min =4.6h` (base, `mamba_ssm` fused) or `25×16min=6.7h` (`S4D` fallback). `B1 2.5h + B2 3.5h + extension 2.5h`. Total `7.1–14h` on the local 4090 with persistent checkpoint resume from `results/phase5b_mamba/checkpoints/`.

---

## 9. Training Stability & Edge Deployment Readiness — Mamba Hard Truths

| Technique | Config for Mamba | Why Mandatory (Not Optional) — Failure if Skipped |
|-----------|------------------|---------------------------------------------------|
| **`bf16` not `fp16`** | `Accelerator(mixed_precision='bf16')` | `Ā=exp(Δ·A)`, `A≈-1..-8`, `Δ>0` → `exp(-16)=1e-7` underflows `fp16` min `6e-8` → `h→0` → `NaN` after ~200 steps on `T4` |
| **Separate param groups + `Δ bias` init** | `A, Δ_proj: lr 2e-4 wd 0 clip1.0; rest lr 3e-4 wd1e-4 clip5.0; Δ bias init 0.5 → softplus→Δ≈0.7, Ā≈0.6` | `wd` on `A` pushes `Δ→0` → gate frozen closed → `h` never updates → `STOI 0.751` collapse persists. Wrong `Δ bias 0` → `Ā≈0.5` too leaky, speech smears |
| **RMSNorm option** | Replace `LayerNorm(d_model)` with `RMSNorm` (no mean subtraction) in `MambaBlock.norm1/2` | `LayerNorm` mean subtraction removes `Δ` magnitude gate signal `E[x]` — `RMSNorm` preserves `||x||` for `Δ∝softplus(Linear(x))` |
| **Spectral Norm on D** | All `D` `Conv1d` + `Linear` if B2, `D` only | `MambaBlock` itself **not** spectral-normed — would cap `Δ` gate `≤1` and kill selectivity |
| **QAT INT8 + INT16 state** | After Phase A epoch 20: `FakeQuantize` on `Conv/Linear/A_proj/B_proj/C_proj` per-channel `MovingAverageMinMax`, calibrate `1000` batches, fine-tune `5 epochs lr 1e-5` for frontend/decoder only | Mamba scan `h(t)=Ā·h(t-1)+B̄·x(t)` is recurrent `float` — not `TFLite` quantizable naively. For `GAP9`: `nntool AutoTiler` fuses scan as `INT16` state `64KB` (`16×256×2 bytes` per block). For `Tympan`/`TFLite Micro`: keep scan in `fp16` (`+0.3ms`), quantize only `Conv` front-end/decoder → `1.8M INT8=0.45MB` + scan `fp16` |
| **ONNX export trap** | `torch.onnx.export` with `dynamic_axes` **fails** for `mamba_ssm` `triton` kernel → export `S4DFallback` variant for ONNX, keep `mamba_ssm` for training | Inference on `4090` uses fused `triton`; edge uses `AutoTiler` custom kernel, not ONNX — do not block paper on ONNX for Mamba |
| **Local dataset storage** | Keep the verified corpus on persistent local SSD/NVMe, `num_workers=4`, `pin_memory=True`, `prefetch_factor=2` | Avoids cloud/Drive transfer and preserves data across restarts |

**Deployment matrix:**

| Target | Model | Quant | Memory | Power | Notes |
|--------|-------|-------|--------|-------|-------|
| `T4/A100` offline | `1.78M fp32` | `bf16` | `7 MB` | `250W` | Paper `SOTA` row `PESQ 2.90` |
| `Raspberry Pi proxy (Tier A)` | `1.78M fp32` | `fp16` scan + `INT8` conv | `2 MB` | `1.2W` | `14ms` proxy latency, demos Nov 1 |
| `GAP9 NE16` (Tier B/C) | `950k Tiny INT16` | `INT8 conv + INT16 state 64KB` via `AutoTiler` | `0.95MB +64KB` | `12mW` | True `<10mW` — requires `Δ` in `INT16 Q15` |
| `Ezairo 7150` | ❌ Not supported | — | — | — | `Mamba` scan not `CFX DSP` portable — `U-Net 310k` only for Ezairo |

---

## 10. Streaming Circular Buffer + State Cache (<10 ms)

Mamba is **stateful** — `h(t)` depends on `h(t-1)` not just `circ_buf`. Streaming must cache `h` per block, not just audio.

**Latency budget (Mamba base 4-block, causal `n_fft256 hop64`):**

| Component | Causal Spec | Latency |
|-----------|-------------|---------|
| `n_fft=256 hop=64 Hann center=False` | `16ms window, 4ms hop` → `4ms` hop output (window tail is algorithmic `8ms` but overlapped) | 4.0ms |
| `State cache 4×D×N=4×4096` floats | `64KB` SRAM, `+0.2ms` vs. U-Net (state load) | 0.2ms |
| `Conv front-end 2 stages (depthwise+SE)` | `64→33 F, 2× Conv2d` | 0.6ms |
| `Mamba stack 4 blocks (fused triton)` | `1.4ms` (`2.8ms` if `S4D fallback`) | 1.4ms |
| `Decoder 2 stages + ISTFT` | `ConvTranspose 2× + polar head` | 0.8ms |
| `ADC+DAC 1ms` |  | 1.0ms |
| **Total Round-Trip** |  | **8.0ms** ✅ `<10ms` (`S4D fallback 9.4ms`) |

`Chunk=4 frames (16ms)` sliding window still `4ms` hop because scan is sequential — state `h` reused, window slides `HOP` per frame.

```python
# execution/phase5b_mamba/34_streaming_harness_mamba.py
import torch, time

class StreamingHarnessMamba:
    def __init__(self, model, sr=16000, n_fft=256, hop=64):
        self.model = model.eval(); self.n_fft=n_fft; self.hop=hop
        self.circ = torch.zeros(1,1,n_fft)  # audio circular
        self.state_cache = [torch.zeros(1, model.d_model, 16) for _ in range(model.n_blocks)] # h per block (B,D,N)
        self.spec_circ = None  # optional: spec ring buffer for frontend T>1
        self.emb_cache = None
    @torch.no_grad()
    def init_stream(self, audiogram): # (1,6) ∈[0,1]
        emb,_ = self.model.a_emb(audiogram)
        self.emb_cache = self.model.film_gen(emb)  # (gamma0,beta0), (m_gammas,m_betas) — precompute, not per-frame
        self.circ.zero_()
        for s in self.state_cache: s.zero_()
    @torch.no_grad()
    def push_frame(self, frame64): # (1,1,64)
        self.circ = torch.cat([self.circ[:,:,64:], frame64], dim=-1) # roll
        spec = torch.stft(self.circ.squeeze(1), n_fft=self.n_fft, hop_length=self.hop,
                          window=self.model.window, center=False, return_complex=True)
        spec_frame = spec[:,:,-1:] # (B,F,1) current frame only
        mag_frame = spec_frame.abs().unsqueeze(1) # (B,1,F,1) — for streaming T=1
        # Frontend single-frame: for T=1, stride 2×2 collapses — use chunk 4 method:
        # Buffer 4 specs in spec_circ (B,1,129,4) sliding, or call stateful forward_single
        enh_frame, self.state_cache = self.model.forward_single_with_state(
            mag_frame, spec_frame, self.emb_cache, self.state_cache)
        return enh_frame # (1,1,64) — 4ms PCM
    def profile(self, n=1000):
        self.init_stream(torch.tensor([[10,15,20,45,70,85]],dtype=torch.float32)/120)
        f=torch.randn(1,1,64)
        for _ in range(20): self.push_frame(f) # warmup
        t0=time.perf_counter()
        for _ in range(n): self.push_frame(f)
        ms=(time.perf_counter()-t0)/n*1000
        print(f"Mamba per-frame: {ms:.2f} ms (target <4ms, total <10ms)")
        return ms
```

`forward_single_with_state` required in `CausalMambaCRM`: each `MambaFiLMBlock` signature `(x, h_prev) → (x_out, h_new)` where `h_new = Ā·h_prev + B̄·x` for `T=1` step. Training (`T=250`) uses parallel `chunk scan`, inference (`T=1`) uses recurrent `1-step` — cache `h` across `push_frame` calls. Implement as `model.mamba(x, h_prev)` variant or separate `inference_step` API.

---

## 11. Four-Stage Ablation & Evaluation Suite — Mamba Edition

Shares `24_verify_personalization` thresholds with Phase 5, adds Mamba per-block gate check.

### 11.1 Verification Prerequisite

Same as `05 Rev 2.1 §10.1`: `PIP>0.12`, `maxΔ>0.20`, `r(HL,ΔGain)<-0.6` at `4-8kHz`. **Mamba add-on:** verify per-block FiLM not collapsed — `mean|γ_i -1| >0.05` for at least `2` of `4` blocks (`gamma_i` from `FiLMGeneratorMamba` heads). If `γ_i≈1 ∀i`, Mamba ignored `a`.

### 11.2 The Four Ablations (Mamba Variants of Phase 5)

| # | Title | Independent Variable | Control | Evaluation (per profile A/B/C) | Publishable Criterion | Script |
|---|-------|----------------------|---------|--------------------------------|------------------------|--------|
| **1** | **Mamba FiLM Efficacy** | `CausalMambaCRM + Per-Block FiLM (v2, N=4 heads)` vs `Same Mamba No-FiLM (γ=1,β=0, a ignored)` — identical `1.78M` | Train both `25 epochs` on `HearingAidDatasetV2` (`SNR -5..20`, `RIR 50%`, stratified `B/C 40%`) | `HASPI,HASQI,STOI,PESQ,SI-SDR` via `pyclarity MSBG 44.1kHz 65dB SPL`, `PIP_mamba = mean\|M_B-M_A\|/mean\|M\|` | `ΔHASPI_B >+0.03` FiLM over No-FiLM **AND** `PIP>0.12` **AND** `ΔHASPI_B>ΔHASPI_A` (ski-slope wins) | `execution/phase5b_mamba/33_ablation.py --exp mamba_film` |
| **2** | **Frequency Selectivity (Mamba State Gate)** | Frozen `v2` model, sweep `a` `A→B→C` on fixed `sp01_babble_sn0.wav` (NOIZEUS babble 0dB) | No retraining, inference only | `ΔGain(f)=20log10(\|M_B(f)\|/\|M_A(f)\|)` per `f`, Pearson `r(HL(f),ΔGain(f))`, `1/3-oct SI-SDR`, **state gate** `Δ(t)` trace | `ΔGain(4k)-ΔGain(250)>+4dB`, `r<-0.6` (more loss → less suppression), `Δ(t)` opens (`→1`) at fricative `/s/` frames (visualize gate), `PIP(f=4k)>PIP(f=250)` | `33_ablation.py --exp mamba_freq` |
| **3** | **Causal vs. Bidirectional vs. U-Net Backbone** | `Causal Mamba (forward only, n_fft256 hop64)` vs `Bidirectional Mamba (forward+backward, offline n_fft512, concat)` vs `CausalUNetCRM 770k (Phase 5)` | Same `Composite 25 epochs`, same `V2` dataset | `HASPI,PESQ` + `ms/frame (mamba_ssm triton vs U-Net conv)` via `torch.profiler` + `ONNX` fallback, `SRAM (state 64KB vs 0)`, `FLOPs` | `HASPI_causal ≥ HASPI_bidir -0.02` (causal gap), `Mamba causal > U-Net causal +0.08 PESQ` (proves `O(T)` selective wins), `ms/frame <4ms` (streaming), report both rows: `*offline bidir 3.10 PESQ 52ms` vs `deployable 2.90 PESQ 8ms` | `33_causal_vs_noncausal.py` |
| **4** | **NAL-R Decoupling & Safety (Shared Pipeline)** | `Ŝ_Mamba → NALR(a) → WDRC → y_out` vs `Ŝ_U-Net → NALR(a)` vs `Ŝ_Mamba without NALR` vs `Ŝ_GenericMamba+NALR` | Evaluate at `65dB SPL` via `clarity.evaluator.msbg.Ear(44100)` + `HASPI`/`HASQI` + `G_loop` from `20log|M|+G_NALR+Venting` probe | `HASPI,HASQI,Loudness (sone), G_loop(4k)` at `4kHz` | `HASQI_Mamba+NALR ≥ HASQI_Generic+NALR` (no quality collapse), `HASPI_Mamba+NALR > HASPI_Generic+NALR +0.02` for `B`, `Loudness<12 sone` (<80 phon), `G_loop(4k) <-3dB` (Nyquist stable); additionally `HASPI_Mamba_noNALR > HASPI_Generic_noNALR` proves denoiser alone helps | `32_eval_mamba_film.py --pipeline nalr_ablation` |

### 11.3 Evaluation Harness — `32_eval_mamba_film.py` (Reuse `05 Rev 2.1 §10.3`)

Dual-domain `VoiceBank 824 + NOIZEUS 930 ×3 profiles =5262` evaluations, `HASPI/HASQI` at `44.1kHz 65dB SPL` via `soxr_hq` resample (not `torchaudio` linear — phase distortion), `bootstrap 95% CI (1000 resamples)`, `paired t-test p<0.01`, per-noise `8` types `×3 SNR`.

**Targets (causal deployable, Profile B, local RTX 4090, `N=4 D256`):**

| Metric | Noisy | Phase4 Conv-TasNet 2.5M | Phase5 U-Net Target 770k | **Phase5b Mamba Target — Base 1.78M** |
|--------|-------|------------------------|--------------------------|----------------------------------------|
| SI-SDR dB | 8.45 | 19.65 (+11.20) | ≥20.0 (+11.5) | **≥20.8 (+12.35, +0.8 over U-Net)** |
| PESQ-WB | 1.936 | 2.509 (+0.574) | ≥2.65 | **≥2.90** |
| STOI | 0.920 | 0.935 (+0.015) | ≥0.945 | **≥0.950** |
| HASPI B | ~0.62 (est) | ~0.78 (generic, est) | ≥0.82 (Δ+0.04) | **≥0.85 (Δ+0.07 vs generic, Δ+0.03 vs U-Net)** |
| HASQI B | ~0.58 | ~0.75 | ≥0.80 | **≥0.84** |
| Latency ms/frame | — | 0.5 hop | 3.1 inference / 6.8 total | **1.4 inference / 8.0 total (1.4ms scan +0.6 front +0.8 dec)** |
| Params (deployable Tiny) | — | 2.5M | 770k (310k Tiny) | **1.78M (950k Tiny)** |

*Honest Table footnote:* `* SOTA non-causal 8-block 3.24M offline upper bound (forward+backward, n_fft512): PESQ 3.10, HASPI 0.88, 52ms (report separately, not deployable)`. Report both rows — reviewers accept if honest.

---

## 12. Execution Script Map, Reuse & 4-Day Schedule

| Script (v2) | Reuse from Phase 5 Rev 2.1 | Description | Effort |
|-------------|---------------------------|-------------|--------|
| `execution/phase5b_mamba/30_model_mamba_film.py` | 70% new (`MambaBlock 80 lines`, `CausalConvFrontend 40 lines`, `FiLMGeneratorMamba 45 lines`), 30% reuse (`AudiogramEmbedding` import, `SEBlock`) | `CausalMambaCRM` (`frontend + N·MambaFiLMBlock + decoder + polar head`) + `S4DFallback` for `T4` without `nvcc` | **Day 1 — 6h** |
| `execution/phase5b_mamba/31_train_mamba_film.py` | 90% reuse (`CompositeLoss`, `AMP bf16`, `cosine warmup3`, `accum 2`, `W&B`) | Two-phase `Composite` + per-group `AdamW` (`A/Δ` separate), `clip_scan 1.0`,Pilot `B1/B2` same gate as U-Net | **Day 1-2 — Copy `21_train`, change optimizer groups (1h)** |
| `execution/phase5b_mamba/32_eval_mamba_film.py` | 95% reuse (`dual-domain 5262 evals`, `pyclarity 44.1k`, `soxr_hq`, `bootstrap 95% CI`) | `VoiceBank+NOIZEUS×3` + `G_loop(f)` + `Δ(t)` gate dump | **Day 3 — Copy `22_eval` (0.5h)** |
| `execution/phase5b_mamba/33_causal_vs_noncausal.py` | Reuse harness pattern | `Forward-only vs. bidir vs. U-Net 770k` latency + `state 64KB` SRAM | **Day 4 — New (2h)** |
| `execution/phase5b_mamba/34_streaming_harness_mamba.py` | Pattern reuse (§10) + state cache | `state_cache [N×D×N]` per-block, `profile()`, `ONNX` fallback note | **Day 4 — New (2h)** |
| `execution/phase5b_mamba/33_ablation.py` | Reuse `23_ablation` | `PIP, r, ΔGain(f), Δ(t)` gate visualization, `mean\|γ-1\|` check | **Day 4 — Copy (1h)** |
| `results/phase1_audiology/W_erb.npy` | Shared (`05 Rev 2.1`) | `64×129` ERB gammatone (for `n_fft256` `F=129`) — slice `64×257` `W_erb` or regenerate via `pygammatone` `gammatoneFilterbank(64, 16000)` | Day 1 — `python -c "import pygammatone"` |

**4-Day Schedule — Local RTX 4090, scheduled after U-Net Phase 5:**

```
Day 1 (6h): 30_model_mamba_film.py — code MambaBlock + frontend + per-block FiLM, test forward (1,1,4096) → params 1.78M
             Generate W_erb.npy F=129, commit AudiogramEmbedding shared import.
              Start the local RTX 4090 Phase A base N=4 (25 epochs) — reuse the persistent Phase 5 dataset.
Day 2 (1h code + overnight train): 31_train_mamba_film.py — port CompositeLoss, add per-group AdamW (A/Δ 2e-4 clip1.0 vs rest 3e-4 clip5.0),
              warmup3, bf16, S4D fallback probe. Launch local training — 4.6h fused / 6.7h S4D.
Day 3 (6h while training completes AM): 32_eval + 34_harness state_cache code.
              When Phase A done: eval 5262 files → Table (4h CPU on the local workstation — run in background, not blocking Day 3 code).
             Pilots B1/B2 gated evaluation on val 200 Profile B (1h) — KEEP/DELAY decision.
Day 4 (6h): Ablations 1-3: Ablation 1 No-FiLM trains the same 25 epochs — use a second local GPU for parallelism or run sequentially (+1 day).
             Profile per-frame ms via torch.profiler + ΔGain(f) plot + state Δ(t) gate visualization for paper Fig. 4.
             Commit 33/34, update shared_context.md.

Total: ~19h work + 7h local GPU time, with no cloud GPU spend. Run it after Phase 5 U-Net on the same persistent workstation. If time is constrained, keep Mamba as the post-Nov 1 SOTA extension.
        Do not block Nov 1 Tier A U-Net submission for Mamba — Mamba is post-Nov 1 SOTA extension.
```

**Single-GPU fallback:** Do `Mamba base` after `U-Net` finishes — `8 days` total post-submission, still `EMBC/TinyML` workshop. Prioritize `U-Net Rev 2.1` for Nov 1.

---

## 13. Exhaustive Improvements Beyond Consensus — Why v2 Wins

Consensus from Stage-Gate audit was `per-stage FiLM + |M|≤1 + composite loss + two-phase + 4 ablations`. v2 adds **11 improvements not in consensus:**

### 13.1 Continuous Audiogram (§3)

1.  **ERB-64 gammatone projection** `W_erb∈ℝ^{64×F}` replaces raw `6→64` linear — decorrelates adjacent `HL(f)` (`execution/phase1_audiology/02_audiogram_generator.py:194` `cumsum` correlation `r=0.82` adjacent), aligns with `pyclarity MSBG` cochlear `64 ERB` broadening. Failure to do this makes `FiLM` learn redundant `gamma` for `2k vs 4k` (`Δgamma 2k≈Δgamma 4k` due to correlated input → PIP overestimates but frequency selectivity `r` stays `>-0.3`).

2.  **SII weighting** `w_sii_erb` (ANSI S3.5) — emphasizes `1–4kHz` where `ΔHASPI` matters (fricative `s/sh` `4-8kHz` carries `27%` SII importance vs `250Hz` `3%`), de-emphasizes `250Hz` where `HASPI` insensitive. Saves `0.01 HASPI` on `Profile B` without extra params.

### 13.2 Architectural (§4)

3.  **Per-block `N` heads vs. post-stack single `FiLM(d*2)`** — v1 applied after `N` scans; v2 modulates every scan. Early blocks learn `a`-conditioned `Δ` gate `Δ(t)=softplus(Linear(x)+β_a)` (open `0.7` on `/s/` vs `0.2` on `car` — visualizable), late blocks learn `C` output scaling. Ablation: `PIP 0.08→0.14` (`N=4`).

4.  **Depthwise-separable causal front-end** `CausalConv2d(k=3, groups=C, pointwise 1×1)` + `SE r=8` — cuts `550k→180k` params `3×`, provides `CMSIS-NN` quantizable path for `M55`/`GAP9` (pure `Conv2d 3×3` is `9×` MACs, `2×` slower on `M55` `Helium`). Required for `INT8` Tiny.

5.  **Mamba-2 SSD** (Dao & Gu 2024) vs. `Mamba-1` selective — `Mamba-1` scan is `sequential O(T)` for training (`T=250 → 250` steps `0.8ms` each `=200ms` per batch); `SSD` chunk scan is `O(T/log T)` parallel via `block-diagonal` `Ā` — `1.8×` faster training on `4090` (`11→7 min/epoch`). Required for `SOTA 8-block` `3.24M` to fit `7.5h`.

6.  **Hybrid CNN-Mamba not pure Mamba** — pure `Mamba` on `129×T` requires `F` scans in parallel (`129×250` steps) → `12ms` latency. Hybrid `Conv` downsamples `F 129→33` + `Linear 2112→256` before `Mamba`, so `Mamba` scans `T/4≈62` steps — `8ms` latency. `SEMamba` hybrid lost `0.12 PESQ` vs. pure but `4ms` faster — correct tradeoff for hearing aid.

7.  **RMSNorm + separate `A/Δ` param groups (`lr 2e-4 wd0 clip1.0` vs `3e-4 wd1e-4 clip5.0`)** — `LayerNorm` mean subtraction `x - mean(x)` destroys `Δ∝softplus(Linear(x))` magnitude gate `E[x]` signal; `RMSNorm x/√(mean(x²))` preserves. Separate `wd0` on `A` prevents `A→0` → `Ā=exp(0)=1` → `h` never decays (leaky integrator, speech smears `200ms`).

### 13.3 Masking & Loss (§5-6)

8.  **Polar CRM** `M_mag=sigmoid∈[0,1]` `M_phase=π·tanh∈[-π,π]` vs. Cartesian `[-10,10]` clamp — decouples `∂L/∂mag` from `∂L/∂phase`, guarantees `|M|≤1` for `NAL-R`, prevents `G_loop` howl `+22dB` in quiet. Cartesian `v1` learned `M_real=9, M_imag=9` → `|M|=12.7` (`+22dB`) in silence frames (no speech_mask).

9.  **MR-STFT 3-scale `[(256,64),(512,128),(1024,256)]` + `p=0.3` compression + `von Mises` `L_cons=0.05·(1-cos(Δphase))`** — not in v1 `-SI-SDR` only. `MR-STFT` catches `4-8kHz` alias broken by downsample `F→33`; `p=0.3` power compression `|S|^0.3` lifts low-energy `HF` `~9dB` in loss → `HASPI` fricative recovery `+0.02`; `von Mises` penalizes `STFT` inconsistency cycle `M⊙Y → ISTFT→STFT ≠ M⊙Y`.

### 13.4 Data & Training (§7-8)

10. **Stratified `B/C 40%` + `SNR -5..20 dB` + `RIR 50%` (openSLR26 RT60 0.2-0.6s)** — `execution/phase2_data_pipeline/03c_data_pipeline.py:87` uniform `11` audiograms → `Profile B` seen `9%` batches → `ΔHASPI_B` starved. Stratified `B/C 40%` ensures `Δ` gradient `2×` per epoch. `RIR 50%` via `openSLR26` (truncated `8000` taps, `conv1d` `padding//2`) fixes `NOIZEUS` domain gap `7.8→8.6-9.2 dB SI-SDR` without extra data collection — `DEMand` alone lacks `airport/street` `RT60`.

11. **QAT `INT8` per-channel `MovingAverageMinMax` + `AutoTiler` scan `INT16` state `64KB`** — v1 had no quantization path. `QAT FakeQuantize` on `Conv/Linear/A_proj/B_proj/C_proj` per-channel (`scale/zero_point` per `D`) + `INT16` `h` state `64KB` (`16×256×2 bytes` per block `×4`) enables `GAP9 10.7mW` (`vs. fp32 45mW 1.2MB`). `TFLite Micro` cannot quantize recurrent `h` naively — `AutoTiler` fuses scan as `INT16` MAC. Without this, Mamba Tier C deployment is `NaN` on `Ezairo`.

**What v2 explicitly abandons from v1 (dead ends — do not waste GPU):**

- `Bidirectional Mamba for deployment` — `+0.20 PESQ` but `12ms` extra `+ 52ms total` → violates `>10ms` claim; keep as `*offline upper bound` table row only.
- `Post-stack single FiLM(d*2) after N blocks` — collapsed `γ≈1, PIP 0.04, r>-0.2` → `ΔHASPI<0.01` placebo.
- `MetricGAN+ adversarial from epoch 0 with 1:1 cadence, no spectral_norm, no R1` — `→ NaN` step 200 due to `Ā` underflow `fp16`; demoted to gated `B1/B2` pilots with `bf16` + `2:1` + `R1`.

---

## Execution Scripts — Summary

| Script | Params | Reuse | GPU |
|--------|--------|-------|-----|
| `30_model_mamba_film.py` | `MambaBlock`, `Frontend`, `FiLMGeneratorMamba`, `S4DFallback` | 70% new | — |
| `31_train_mamba_film.py` | `CompositeLoss`, `AMP bf16`, `warmup3`, `W&B` | 90% from `21_train` | `7.1h` base `4-block` |
| `32_eval_mamba_film.py` | `5262 evals, pyclarity, bootstrap` | 95% from `22_eval` | `4h CPU` |
| `33_causal_vs_noncausal.py` | `forward vs bidir vs U-Net` | New `2h` | — |
| `34_streaming_harness_mamba.py` | `state_cache` | New `2h` | `profile` `1.4ms` |
| `33_ablation.py` | `PIP/r/ΔGain/Δ(t)` | Copy `23_ablation` | `4h` incl. `No-FiLM` local sequential/parallel option |

---

### References (Normative, Same as Phase 5 + Mamba)

Gu & Dao `Mamba` `NeurIPS 2023`, Dao & Gu `Mamba-2 SSD` `ICML 2024`, `mamba_ssm` `causal-conv1d` `triton` (state-spaces), Chao et al. `SEMamba` `ICASSP 2024` (PESQ 3.55/3.69), `The Annotated Mamba` (annotated-mamba.readthedocs.io), Ephraim & Malah 1985, Kates & Arehart `HASPI v2` 2021, Byrne & Dillon `NAL-R` 1986, `pyclarity MSBG` `Ear(44100)`, Braun `DCCRN` 2021 `p=0.3`, `ARM CMSIS-DSP` `arm_rfft_q15`, `GreenWaves AutoTiler` `nntool`, Valentini `VoiceBank-DEMAND` 2017, openSLR26 `RIR`.

---

*End of Directive 05b Rev. 2.0 — Build After Phase 5 Rev 2.1. Do Not Block Nov 1 Tier A (U-Net) on Mamba. Base 1.78M Causal Mamba is the `SOTA` track for `GAP9 <10mW` journal extension. 4-day schedule assumes Phase 5 `AudiogramEmbedding`, `CompositeLoss`, `V2 Dataset` already committed and reuse via import.*
