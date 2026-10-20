# execution/phase5_unet_film/22_streaming_inference.py
# Approximate frame-step benchmark harness for the Phase A offline model.
# Phase A uses center=True STFT/iSTFT and this helper is not a proof of causal
# streaming equivalence. Do not use its timing as a final hearing-aid latency
# claim until a stateful causal model is implemented and validated.

import torch
import torch.nn.functional as F

class StreamingHarness:
    """Shape-level frame-step harness with cached audiogram conditioning."""
    def __init__(self, model, sr=16000, n_fft=256, hop=64):
        self.model = model.eval()
        self.sr = sr; self.n_fft = n_fft; self.hop = hop
        self.buffer = torch.zeros(1, 1, n_fft)
        self.emb_cache = None

    @torch.no_grad()
    def init_stream(self, audiogram):
        # audiogram (1,6) in [0,1]
        self.emb_cache, _ = self.model.a_emb(audiogram)
        # Precompute FiLM gammas/betas
        self.gammas, self.betas = self.model.film_gen(self.emb_cache)
        self.buffer.zero_()

    @torch.no_grad()
    def push_frame(self, frame64):
        # frame64 (1,1,64) — next hop, returns PCM hop (1,1,64) via overlap-add.
        self.buffer = torch.cat([self.buffer[:, :, 64:], frame64], dim=-1)
        # Center=True STFT on 256 window — T=1 frame for streaming
        spec = torch.stft(self.buffer.squeeze(1), n_fft=self.n_fft, hop_length=self.hop,
                          win_length=self.n_fft, window=self.model.window,
                          center=True, return_complex=True)  # (B,F,T)
        mag = spec.abs().unsqueeze(1)
        skips = self.model.encoder(mag)
        s1, s2, s3, s4 = skips
        bottleneck = self.model.film_layer(s4, self.gammas[3], self.betas[3])
        bottleneck = self.model.bottleneck_attn(bottleneck)
        d4 = self.model.dec4(bottleneck, s3, self.gammas[0], self.betas[0])
        d3 = self.model.dec3(d4, s2, self.gammas[1], self.betas[1])
        d2 = self.model.dec2(d3, s1, self.gammas[2], self.betas[2])
        d2_up = F.interpolate(d2, size=(self.model.n_freq, spec.shape[-1]), mode="bilinear", align_corners=False)
        logits = self.model.final(d2_up)
        m_mag = torch.sigmoid(logits[:, 0])  # (B,F,T)
        m_phase = torch.tanh(logits[:, 1]) * torch.pi
        m_real = m_mag * torch.cos(m_phase)
        m_imag = m_mag * torch.sin(m_phase)
        enh_spec = torch.complex(m_real * spec.real - m_imag * spec.imag,
                                 m_real * spec.imag + m_imag * spec.real)
        enh_wav = torch.istft(enh_spec, n_fft=self.n_fft, hop_length=self.hop,
                              win_length=self.n_fft, window=self.model.window,
                              center=True, length=self.n_fft)
        # Return last hop (64 samples) from the 256-sample ISTFT window
        hop_out = enh_wav[:, -self.hop:]
        return hop_out.unsqueeze(1)  # (1,1,64) PCM

    def profile(self, n=200):
        import time
        self.init_stream(torch.tensor([[10, 15, 20, 45, 70, 85]], dtype=torch.float32) / 120.0)
        frame = torch.randn(1, 1, 64)
        for _ in range(20): self.push_frame(frame)
        t0 = time.perf_counter()
        for _ in range(n): self.push_frame(frame)
        ms = (time.perf_counter() - t0) / n * 1000
        print(f"Phase A frame-step benchmark: {ms:.2f} ms (not a causal latency claim)")
        return ms
