# Directive 05 — U-Net + Self-Attention + FiLM (Core Contribution)

**Phase:** 5  
**Goal:** Build our core personalised deep learning model — a U-Net with self-attention, FiLM (Feature-wise Linear Modulation) audiogram conditioning, complex ratio masking, and MetricGAN+ perceptual training.  
**Estimated time:** 2 weeks (Oct 5 – Oct 18, 2026)  
**Lead:** Both  
**Output:** The personalized denoiser that proves audiogram conditioning improves HASPI over the generic 1D CNN.

---

## Why This Is Our Core Contribution

Every improvement in this model over Model 3 (1D CNN) must be attributable to **audiogram personalisation**:

- The 1D CNN gets a single, generic enhancement for everyone
- This model gets a **different enhancement for every audiogram**
- If HASPI improves — especially for extreme audiograms (severe high-frequency loss) — the experiment succeeds

The combination of **FiLM conditioning + complex masking + MetricGAN+ with HASPI discriminator** has not been published before. This is the novelty.

---

## Architecture Overview

```
INPUT:
  noisy_spectrogram  Y ∈ ℂ^{F × T}   (complex STFT: F=257 freq bins, T=frames)
  audiogram vector   a ∈ ℝ^6          normalised to [0, 1]

PROCESSING:
  1. Split Y into magnitude |Y| and phase ∠Y
  2. Feed |Y| through U-Net encoder (4 stages, downsampling)
  3. At the bottleneck: apply Self-Attention (global context)
  4. Apply FiLM conditioning using audiogram a (personalise)
  5. Decoder (4 stages, upsampling + skip connections from encoder)
  6. Output head: 2 × F channels → Complex Ratio Mask (CRM) [M_real, M_imag]

RECONSTRUCTION:
  enhanced_real = M_real × Y_real - M_imag × Y_imag
  enhanced_imag = M_real × Y_imag + M_imag × Y_real
  enhanced_spec = enhanced_real + j × enhanced_imag
  enhanced_wav  = ISTFT(enhanced_spec)
```

---

## Component 1: The U-Net Backbone

The U-Net has 4 encoder stages and 4 decoder stages connected by skip connections.

```python
# Encoder: each stage halves the time-frequency dimensions
Encoder Stage 1: Conv2D(1,   32, kernel=3) + BatchNorm + LeakyReLU + stride=2
Encoder Stage 2: Conv2D(32,  64, kernel=3) + BatchNorm + LeakyReLU + stride=2
Encoder Stage 3: Conv2D(64,  128, kernel=3) + BatchNorm + LeakyReLU + stride=2
Encoder Stage 4: Conv2D(128, 256, kernel=3) + BatchNorm + LeakyReLU + stride=2

# Bottleneck (after Encoder 4, before FiLM and Attention):
# Feature map shape: (B, 256, F/16, T/16)

# Decoder: each stage doubles dimensions, concatenates skip connection
Decoder Stage 4: ConvTranspose2D(256+256, 128) + BatchNorm + ReLU  ← skip from Encoder 4
Decoder Stage 3: ConvTranspose2D(128+128,  64) + BatchNorm + ReLU  ← skip from Encoder 3
Decoder Stage 2: ConvTranspose2D( 64+ 64,  32) + BatchNorm + ReLU  ← skip from Encoder 2
Decoder Stage 1: ConvTranspose2D( 32+ 32,   2)                     ← skip from Encoder 1

# Output: 2 channels = [M_real, M_imag] — the Complex Ratio Mask
```

**Why skip connections?** The bottleneck compresses everything into an abstract representation — fine details are lost. Skip connections copy the encoder's detailed feature maps directly to the decoder, so the decoder can use both abstract understanding AND fine-grained detail.

---

## Component 2: Self-Attention in the Bottleneck

After the encoder compresses the spectrogram, we add a self-attention layer. This lets every position in the compressed feature map attend to (look at) every other position.

**Why it's placed at the bottleneck:** The feature maps are smallest here (most compressed), so the attention computation is cheapest. It adds global context before the FiLM conditioning applies.

```python
class BottleneckAttention(nn.Module):
    """1D Self-Attention applied to the bottleneck feature map."""
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv1d(channels, channels // 8, 1)
        self.key   = nn.Conv1d(channels, channels // 8, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))  # starts at 0 (residual)
    
    def forward(self, x):
        # x shape: (B, C, H, W) — 4D bottleneck feature map
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W)  # flatten spatial dims: (B, C, H×W)
        
        q = self.query(x_flat)           # (B, C//8, H×W)
        k = self.key(x_flat)             # (B, C//8, H×W)
        v = self.value(x_flat)           # (B, C,    H×W)
        
        # Attention scores: how much each position attends to each other
        attn = torch.bmm(q.permute(0, 2, 1), k)   # (B, H×W, H×W)
        attn = F.softmax(attn / (C // 8) ** 0.5, dim=-1)  # scaled softmax
        
        out = torch.bmm(v, attn.permute(0, 2, 1))  # (B, C, H×W)
        out = out.view(B, C, H, W)                  # restore 4D shape
        
        # Residual: start at 0 (identity), gradually learn to use attention
        return self.gamma * out + x
```

---

## Component 3: FiLM Conditioning (The Personalisation)

**FiLM** stands for **Feature-wise Linear Modulation**. A small auxiliary network takes the audiogram vector and outputs scale (γ, "gamma") and shift (β, "beta") parameters that modulate the bottleneck features.

```python
class FiLMGenerator(nn.Module):
    """Maps audiogram vector → scale and shift for bottleneck features."""
    def __init__(self, audiogram_dim=6, feature_channels=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(audiogram_dim, 64),   # 6 → 64
            nn.ReLU(),
            nn.Linear(64, 128),             # 64 → 128
            nn.ReLU(),
            nn.Linear(128, feature_channels * 2)  # → 512 (256 γ + 256 β)
        )
        self.feature_channels = feature_channels
    
    def forward(self, audiogram):
        # audiogram: (B, 6) — batch of normalised audiogram vectors
        params = self.mlp(audiogram)                           # (B, 512)
        gamma = params[:, :self.feature_channels]              # (B, 256)
        beta  = params[:, self.feature_channels:]              # (B, 256)
        return gamma, beta


class FiLMLayer(nn.Module):
    """Applies FiLM modulation to a 4D feature map."""
    def forward(self, features, gamma, beta):
        # features: (B, C, H, W)
        # gamma, beta: (B, C) — need to broadcast over H and W
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        return gamma * features + beta             # broadcast to (B, C, H, W)
```

**Application order in the forward pass:**
```
encoder_output → self_attention → film_layer(gamma, beta) → decoder_input
```

---

## Component 4: Complex Ratio Mask (CRM) Prediction

The decoder's final layer outputs 2 channels: `M_real` and `M_imag`. These are NOT passed through a Sigmoid (unlike IRM), because the complex mask can have values outside [0, 1].

Instead, clamp the values to avoid explosion:

```python
def apply_crm(mask_real, mask_imag, noisy_real, noisy_imag):
    """Apply Complex Ratio Mask to complex STFT."""
    # Clamping keeps values stable during training
    mask_real = torch.clamp(mask_real, -10, 10)
    mask_imag = torch.clamp(mask_imag, -10, 10)
    
    # Complex multiplication: (M_real + j×M_imag) × (Y_real + j×Y_imag)
    enhanced_real = mask_real * noisy_real - mask_imag * noisy_imag
    enhanced_imag = mask_real * noisy_imag + mask_imag * noisy_real
    
    return enhanced_real, enhanced_imag
```

---

## Component 5: MetricGAN+ Training with HASPI Discriminator

**Standard training** uses SI-SDR as the loss — it measures waveform accuracy, not perceptual quality for hearing-impaired listeners.

**MetricGAN+ training** adds a discriminator that learns to predict the HASPI score. The generator (our U-Net) is trained adversarially to maximise the predicted HASPI.

**Our novel contribution:** Most MetricGAN+ implementations use PESQ as the target metric. We use **HASPI** — which takes the audiogram as input and measures hearing-aid-specific intelligibility. This combination has not been published.

```python
# Training loop (simplified):

# Generator forward pass:
enhanced_audio = generator(noisy_audio, audiogram)

# Primary loss (SI-SDR):
primary_loss = -si_sdr(enhanced_audio, clean_audio)

# Discriminator predicts HASPI score:
predicted_haspi = discriminator(enhanced_audio, clean_audio, audiogram)
real_haspi = compute_haspi(enhanced_audio, clean_audio, audiogram)  # pyclarity

# Discriminator loss (make it accurate):
disc_loss = F.mse_loss(predicted_haspi, real_haspi)

# Generator adversarial loss (fool discriminator into predicting high HASPI):
gen_adv_loss = -predicted_haspi.mean()

# Total generator loss:
total_loss = primary_loss + 0.1 * gen_adv_loss

# Training:
# Step 1: Update discriminator (disc_loss.backward())
# Step 2: Update generator (total_loss.backward())
```

**Note:** Computing real HASPI during training is slow. Batch it every 5 steps, or train in two phases: first with SI-SDR only, then fine-tune with the MetricGAN+ discriminator.

---

## Verification Test (Critical)

This test proves the model actually uses the audiogram (not ignoring it):

```python
# execution/18_verify_personalization.py
# Take ONE noisy clip and run it through the model with DIFFERENT audiograms
# If the output mask is different for each audiogram, FiLM is working
# If the output mask is identical, the model is ignoring the audiogram — debug gradients

noisy = load_audio("test_clip.wav")  # one noisy clip

audiogram_normal   = [10, 10, 10, 10, 10, 10]   # normal hearing
audiogram_hf_loss  = [10, 15, 20, 45, 70, 85]   # severe high-frequency loss
audiogram_flat_loss = [60, 60, 60, 60, 60, 60]  # flat severe loss

mask_normal   = model(noisy, audiogram_normal)
mask_hf_loss  = model(noisy, audiogram_hf_loss)
mask_flat_loss = model(noisy, audiogram_flat_loss)

# These must be numerically different (check max absolute difference)
assert torch.max(torch.abs(mask_normal - mask_hf_loss)) > 0.01
print("✅ Personalisation verified — masks differ per audiogram")
```

---

## Execution Scripts

| Script | What It Does |
|--------|-------------|
| `execution/13_unet_components.py` | Defines all components: U-Net, BottleneckAttention, FiLMGenerator, FiLMLayer |
| `execution/14_model_unet_film.py` | Assembles the full PersonalizedDenoiser model |
| `execution/15_train_unet_film.py` | Two-phase training: Phase A (SI-SDR), Phase B (+ MetricGAN+ with HASPI discriminator) |
| `execution/16_eval_unet_film.py` | Evaluation on NOIZEUS: HASPI, HASQI, STOI, SI-SDR per audiogram profile |
| `execution/17_ablation.py` | Ablation study: compare model with/without attention, with/without FiLM |
| `execution/18_verify_personalization.py` | The critical verification that different audiograms produce different masks |

---

## Resources

- **Paper:** Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015) — original 8-page paper, very readable
- **Paper:** Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer" (2017) — Sections 1-4
- **Paper:** Hu et al., "DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement" (2020) — shows complex masking in a U-Net
- **Paper:** Fu et al., "MetricGAN+: An Improved Version of MetricGAN for SE" (2021) — for the discriminator training
- **Code:** SpeechBrain has MetricGAN+ built in — use as reference for the discriminator
- **Video:** YouTube — "U-Net architecture explained" (~15 min)
- **Video:** YouTube — "GAN (Generative Adversarial Network) training explained" (~20 min)

---

## Success Criteria

1. `execution/18_verify_personalization.py` passes — different audiograms → different masks
2. HASPI improvement over Model 3 (1D CNN) is **larger** than STOI improvement
   - (This proves the improvement is specifically due to hearing-loss-aware personalisation)
3. HASPI improvement is **largest** for the severe high-frequency loss audiogram `[10,15,20,45,70,85]`

---

## Learnings Log

*(Agent: append findings here — what learning rate for MetricGAN+, any mode collapse symptoms, FiLM ablation results)*
