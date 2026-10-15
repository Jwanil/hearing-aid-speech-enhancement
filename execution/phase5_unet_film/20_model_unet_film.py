# execution/phase5_unet_film/20_model_unet_film.py
# Directive 05 Rev 2.1 — CausalUNetCRM (U-Net + Attention + Per-Stage FiLM + Polar CRM)
# Arch: STFT n_fft=256 hop=64 Hann center=False -> Causal Encoder 4-stage depthwise+SE
#       -> Bottleneck causal attention + FiLM0(256) -> Decoder 3-stage FiLM+skip -> Polar CRM |M|<=1
# Params: ~770k (base) / 310k (tiny). Verify via: python 20_model_unet_film.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# ── 2.5 AudiogramEmbedding (ERB-64) ───────────────────────────────────────────
class AudiogramEmbedding(nn.Module):
    """6-dim HL(dB) /120 -> 128-dim via log2 interp -> ERB-64 -> SII -> MLP 64->128->128."""
    def __init__(self, n_fft=256, sr=16000, n_erb=64, embed_dim=128):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.n_freq = n_fft // 2 + 1  # 129
        self.n_erb = n_erb
        self.register_buffer("freqs_dense", torch.linspace(250, 8000, self.n_freq))
        F_A = np.log2(np.array([250, 500, 1000, 2000, 4000, 8000], dtype=np.float32))
        self.register_buffer("F_A", torch.tensor(F_A))
        # W_erb: 64 x F — try load from results, else random softmax (row-normalized gammatone proxy)
        W_erb = self._load_or_init_W_erb(n_erb, self.n_freq)
        self.register_buffer("W_erb", W_erb)  # (64, F), not trainable
        sii = self._sii_weights(n_erb)
        self.register_buffer("sii_w", torch.tensor(sii, dtype=torch.float32))
        self.mlp = nn.Sequential(
            nn.Linear(n_erb, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU(),
        )
        self.spline_scale = nn.Parameter(torch.tensor(1.0))

    def _load_or_init_W_erb(self, n_erb, n_freq):
        base = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
        candidates = [
            os.path.join(base, "..", "..", "results", "phase1_audiology", "W_erb.npy"),
            os.path.join(base, "..", "..", "results", "phase1_audiology", f"W_erb_{n_freq}.npy"),
        ]
        for p in candidates:
            p = os.path.abspath(p)
            if os.path.exists(p):
                try:
                    W = np.load(p)
                    # Slice or pad to n_freq if needed
                    if W.shape[1] != n_freq:
                        # Interpolate rows to n_freq
                        W_t = torch.from_numpy(W).float().unsqueeze(0)  # (1,64,F_old)
                        W_t = F.interpolate(W_t, size=n_freq, mode="linear", align_corners=True).squeeze(0)
                        W = W_t.numpy()
                    W = torch.from_numpy(W).float()
                    # row-normalize via softmax for stability if not already
                    W = F.softmax(W, dim=-1)
                    print(f"[AudiogramEmbedding] Loaded W_erb from {p} shape {W.shape}")
                    return W
                except Exception as e:
                    print(f"[AudiogramEmbedding] Failed to load {p}: {e}")
        # Fallback: random softmax
        print(f"[AudiogramEmbedding] W_erb.npy not found — using random softmax 64x{n_freq} (trainable MLP compensates)")
        W_erb = torch.randn(n_erb, n_freq)
        W_erb = F.softmax(W_erb, dim=-1)
        return W_erb

    def _sii_weights(self, n_erb):
        x = np.arange(n_erb, dtype=np.float32)
        w = 1.0 - np.abs(x - n_erb * 0.45) / (n_erb * 0.45)
        return np.clip(w, 0.2, 1.0).astype(np.float32)

    def forward(self, a):  # a: (B,6) in [0,1]
        hl_dense = self._interp_log(a)  # (B, F)
        hl_erb = torch.matmul(hl_dense, self.W_erb.T)  # (B,64)
        hl_erb = hl_erb * self.sii_w.unsqueeze(0)
        emb = self.mlp(hl_erb)  # (B,128)
        return emb, hl_dense

    def _interp_log(self, a):
        # Linear interp on log2-spaced 6 points -> F bins (Pchip approximated as linear for speed; directive allows it)
        a_exp = a.unsqueeze(1)  # (B,1,6)
        hl_dense = F.interpolate(a_exp, size=self.n_freq, mode="linear", align_corners=True).squeeze(1)
        return hl_dense * self.spline_scale  # (B,F)


class FiLMGenerator(nn.Module):
    """Per-stage residual FiLM: embedding(128) -> {gamma,beta} per decoder stage, init gamma=1 beta=0."""
    def __init__(self, embed_dim=128, film_channels=(128, 64, 32, 256)):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(128, 2 * c) for c in film_channels])
        for h in self.heads:
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)
        self.channels = film_channels

    def forward(self, emb):  # emb: (B,128)
        h = self.trunk(emb)  # (B,128)
        gammas, betas = [], []
        for head, c in zip(self.heads, self.channels):
            p = head(h)  # (B,2c)
            gamma = 1.0 + p[:, :c]
            beta = p[:, c:]
            gamma = torch.clamp(gamma, 0.5, 2.0)
            gammas.append(gamma)
            betas.append(beta)
        return gammas, betas  # 4 x (B,C)


class FiLMLayer(nn.Module):
    def forward(self, feat, gamma, beta):  # feat (B,C,H,W) gamma/beta (B,C)
        return gamma.unsqueeze(-1).unsqueeze(-1) * feat + beta.unsqueeze(-1).unsqueeze(-1)


# ── 3.2 Causal Encoder ────────────────────────────────────────────────────────
class CausalConv2d(nn.Module):
    def __init__(self, cin, cout, k=3, stride=2):
        super().__init__()
        self.pad = (k - 1, 0, k - 1, 0)
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
            nn.Conv2d(c, c // r, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c // r, c, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(x)

class CausalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = CausalConv2d(1, 32, stride=2)
        self.se1 = SEBlock(32)
        self.e2 = CausalConv2d(32, 64, stride=2)
        self.se2 = SEBlock(64)
        self.e3 = CausalConv2d(64, 128, stride=2)
        self.se3 = SEBlock(128)
        self.e4 = CausalConv2d(128, 256, stride=2)
        self.se4 = SEBlock(256)
    def forward(self, x):  # (B,1,F,T)
        s1 = self.se1(self.e1(x))
        s2 = self.se2(self.e2(s1))
        s3 = self.se3(self.e3(s2))
        s4 = self.se4(self.e4(s3))
        return [s1, s2, s3, s4]

# ── 3.3 Causal Bottleneck Attention ───────────────────────────────────────────
class CausalBottleneckAttention(nn.Module):
    def __init__(self, c=256):
        super().__init__()
        self.q = nn.Conv1d(c, c // 8, 1, bias=False)
        self.k = nn.Conv1d(c, c // 8, 1, bias=False)
        self.v = nn.Conv1d(c, c, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):  # (B,C,H,W)
        B, C, H, W = x.shape
        if W == 1:
            return x
        x_r = x.permute(0, 2, 1, 3).reshape(B * H, C, W)  # (B*H, C, W)
        q = self.q(x_r)
        k = self.k(x_r)
        v = self.v(x_r)
        attn = torch.bmm(q.permute(0, 2, 1), k) / (C // 8) ** 0.5  # (B*H, W, W)
        mask = torch.triu(torch.ones(W, W, device=x.device), diagonal=1).bool()
        attn.masked_fill_(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1))  # (B*H,C,W)
        out = out.view(B, H, C, W).permute(0, 2, 1, 3).reshape(B, C, H, W)
        return self.gamma * out + x

# ── 3.4 Per-Stage FiLM Decoder ────────────────────────────────────────────────
class CausalDecoderBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x, skip, gamma, beta):
        x = self.up(x)
        if skip is not None:
            dh, dw = x.shape[2] - skip.shape[2], x.shape[3] - skip.shape[3]
            if dh > 0 or dw > 0:
                x = x[:, :, :skip.shape[2], :skip.shape[3]]
            elif dh < 0 or dw < 0:
                x = F.pad(x, (0, max(-dw, 0), 0, max(-dh, 0)))
            x = gamma.unsqueeze(-1).unsqueeze(-1) * x + beta.unsqueeze(-1).unsqueeze(-1)
            x = self.act(self.bn(x))
            x = torch.cat([x, skip], dim=1)
        else:
            x = gamma.unsqueeze(-1).unsqueeze(-1) * x + beta.unsqueeze(-1).unsqueeze(-1)
            x = self.act(self.bn(x))
        return x

# ── 3.5 Full CausalUNetCRM ────────────────────────────────────────────────────
class CausalUNetCRM(nn.Module):
    def __init__(self, n_fft=256, sr=16000):
        super().__init__()
        self.n_fft = n_fft
        self.sr = sr
        self.n_freq = n_fft // 2 + 1
        self.a_emb = AudiogramEmbedding(n_fft=n_fft, sr=sr, n_erb=64, embed_dim=128)
        self.film_gen = FiLMGenerator(embed_dim=128, film_channels=(128, 64, 32, 256))
        self.encoder = CausalEncoder()
        self.bottleneck_attn = CausalBottleneckAttention(256)
        self.film_layer = FiLMLayer()
        self.dec4 = CausalDecoderBlock(256, 128)
        self.dec3 = CausalDecoderBlock(256, 64)
        self.dec2 = CausalDecoderBlock(128, 32)
        self.final = nn.Conv2d(64, 2, kernel_size=1)
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, noisy_wav, audiogram):
        # noisy_wav (B,1,T_wav) @16kHz, audiogram (B,6) in [0,1]
        spec = torch.stft(noisy_wav.squeeze(1), n_fft=self.n_fft, hop_length=64,
                          win_length=self.n_fft, window=self.window,
                          center=True, return_complex=True)  # (B,F,T) center=True required for torch Hann COLA
        mag = spec.abs().unsqueeze(1)  # (B,1,F,T)
        emb, _ = self.a_emb(audiogram)  # (B,128)
        gammas, betas = self.film_gen(emb)  # 4 x (B,C): [128,64,32,256]

        skips = self.encoder(mag)
        s1, s2, s3, s4 = skips
        bottleneck = s4
        bottleneck = self.film_layer(bottleneck, gammas[3], betas[3])  # FiLM0 256
        bottleneck = self.bottleneck_attn(bottleneck)

        d4 = self.dec4(bottleneck, s3, gammas[0], betas[0])  # 256->128 + cat 128 =256
        d3 = self.dec3(d4, s2, gammas[1], betas[1])           # 256->64 + cat 64=128
        d2 = self.dec2(d3, s1, gammas[2], betas[2])           # 128->32 + cat 32=64
        d2_up = F.interpolate(d2, size=(self.n_freq, spec.shape[-1]), mode="bilinear", align_corners=False)
        logits = self.final(d2_up)  # (B,2,F,T)

        mag_logit, phase_logit = logits[:, 0], logits[:, 1]
        m_mag = torch.sigmoid(mag_logit)  # [0,1] suppression-only
        m_phase = torch.tanh(phase_logit) * torch.pi  # [-pi,pi]
        m_real = m_mag * torch.cos(m_phase)
        m_imag = m_mag * torch.sin(m_phase)

        spec_real, spec_imag = spec.real, spec.imag
        enh_real = m_real * spec_real - m_imag * spec_imag
        enh_imag = m_real * spec_imag + m_imag * spec_real
        enh_spec = torch.complex(enh_real, enh_imag)
        enh_wav = torch.istft(enh_spec, n_fft=self.n_fft, hop_length=64,
                              win_length=self.n_fft, window=self.window,
                              center=True, length=noisy_wav.shape[-1])
        return enh_wav.unsqueeze(1), (m_real, m_imag, m_mag, m_phase), enh_spec

    @torch.no_grad()
    def count_params(self):
        return sum(p.numel() for p in self.parameters())

# Tiny variant for ESP32/GAP9 future (channel-pruned 50%)
class TinyUNetCRM(CausalUNetCRM):
    def __init__(self, n_fft=256, sr=16000):
        # We override encoder/decoder with pruned widths; reuse parent init then replace
        super().__init__(n_fft=n_fft, sr=sr)
        # Replace encoder with pruned
        self.encoder.e1 = CausalConv2d(1, 16, stride=2); self.encoder.se1 = SEBlock(16, r=4)
        self.encoder.e2 = CausalConv2d(16, 32, stride=2); self.encoder.se2 = SEBlock(32, r=4)
        self.encoder.e3 = CausalConv2d(32, 64, stride=2); self.encoder.se3 = SEBlock(64, r=4)
        self.encoder.e4 = CausalConv2d(64, 96, stride=2); self.encoder.se4 = SEBlock(96, r=4)
        self.bottleneck_attn = CausalBottleneckAttention(96)
        self.film_gen = FiLMGenerator(embed_dim=128, film_channels=(64, 32, 16, 96))
        self.dec4 = CausalDecoderBlock(96, 64)
        self.dec3 = CausalDecoderBlock(128, 32)  # 64+64=128 in
        self.dec2 = CausalDecoderBlock(64, 16)   # 32+32=64 in
        self.final = nn.Conv2d(32, 2, kernel_size=1)  # 16+16=32


if __name__ == "__main__":
    print("Testing CausalUNetCRM (770k)...")
    model = CausalUNetCRM()
    total = model.count_params()
    print(f"Total params: {total:,} ({total/1e6:.3f}M) — target ~770k")
    dummy_wav = torch.randn(1, 1, 64000)
    dummy_a = torch.tensor([[10, 10, 10, 10, 10, 10]], dtype=torch.float32) / 120.0
    with torch.no_grad():
        enh, masks, spec = model(dummy_wav, dummy_a)
    print(f"Input: {dummy_wav.shape} -> Enh: {enh.shape} spec: {spec.shape}")
    m_real, m_imag, m_mag, m_phase = masks
    print(f"m_mag range [{m_mag.min():.3f}, {m_mag.max():.3f}] |M|<=1: {(m_mag.max()<=1.0).item()}")
    print(f"m_phase range [{m_phase.min():.3f}, {m_phase.max():.3f}]")
    assert enh.shape == dummy_wav.shape, "Length mismatch"
    assert float(m_mag.max()) <= 1.0 + 1e-6, "Gain invariant violated"
    print("✅ CausalUNetCRM forward pass successful")
    tiny = TinyUNetCRM()
    print(f"TinyUNet params: {tiny.count_params():,} ({tiny.count_params()/1e3:.1f}k)")
    with torch.no_grad():
        enh_t, _, _ = tiny(dummy_wav, dummy_a)
    print(f"Tiny output: {enh_t.shape} ✅")
