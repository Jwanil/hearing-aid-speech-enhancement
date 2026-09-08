from __future__ import annotations
"""
Phase 0 — DSP Fundamentals: STFT Visualisation
================================================
Directive: directives/00_dsp_fundamentals.md

What this script does:
  1. Loads a .wav speech file (or synthesises a test tone if none found)
  2. Plots the raw waveform
  3. Computes the STFT (Short-Time Fourier Transform) and plots its spectrogram
  4. Demonstrates high-frequency removal: zeroes out bins above a cutoff, inverts
     back to audio with ISTFT, and saves the degraded .wav so you can hear the effect
  5. Overlays the magnitude spectrum before/after removal

Usage:
    python execution/01_stft_visualize.py
    python execution/01_stft_visualize.py --wav path/to/your.wav

Outputs:
    results/plots/00_waveform.png
    results/plots/00_spectrogram.png
    results/plots/00_hf_removal_spectrum.png
    results/audio_demos/00_original.wav
    results/audio_demos/00_hf_removed.wav
"""

import argparse
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — works without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(ROOT, "results", "plots")
DEMO_DIR = os.path.join(ROOT, "results", "audio_demos")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DEMO_DIR, exist_ok=True)

# ─── STFT Parameters ─────────────────────────────────────────────────────────
# These values are standard for 16 kHz speech processing.
# n_fft = 512  →  257 unique frequency bins (0 Hz to 8 kHz)
# hop_length = 128  →  8 ms time step between frames (128/16000)
# win_length = 512  →  32 ms analysis window — long enough to resolve low freqs
N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512
TARGET_SR = 16_000   # 16 kHz — standard for speech enhancement models

# ─── High-frequency removal cutoff (simulates high-frequency hearing loss) ───
HF_CUTOFF_HZ = 3_000   # Remove everything above 3 kHz


# ─── Load audio ────────────────────────────────────────────────────────────────
def load_audio(wav_path: str | None) -> tuple[torch.Tensor, int]:
    """Load a .wav file (defaults to NOIZEUS clean sp01 if not provided) and mix to mono."""
    if not wav_path:
        wav_path = os.path.join(ROOT, "data", "processed", "clean", "noizeus", "clean", "sp01.wav")
        
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio file not found: {wav_path}. Please run Phase 2 data standardisation first.")
        
    waveform, sr = torchaudio.load(wav_path)
    print(f"Loaded: {wav_path}  |  SR={sr} Hz  |  shape={waveform.shape}")

    # Resample to target SR if needed
    if sr != TARGET_SR:
        resampler = T.Resample(sr, TARGET_SR)
        waveform = resampler(waveform)
        sr = TARGET_SR
        print(f"   Resampled to {TARGET_SR} Hz")

    # Mix to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform, sr


# ─── STFT helpers ──────────────────────────────────────────────────────────────
def compute_stft(waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (magnitude, phase) both shape (F, T)."""
    stft_fn = T.Spectrogram(
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        power=None,          # complex output
        window_fn=torch.hann_window,
    )
    complex_spec = stft_fn(waveform.squeeze(0))   # (F, T) complex
    magnitude = complex_spec.abs()
    phase = complex_spec.angle()
    return magnitude, phase


def hz_to_bin(hz: float, sr: int, n_fft: int) -> int:
    """Convert a frequency in Hz to the nearest STFT bin index."""
    return int(np.round(hz * n_fft / sr))


# ─── Plotting ──────────────────────────────────────────────────────────────────
def plot_waveform(waveform: torch.Tensor, sr: int):
    fig, ax = plt.subplots(figsize=(10, 3))
    t = np.linspace(0, waveform.shape[-1] / sr, waveform.shape[-1])
    ax.plot(t, waveform.squeeze().numpy(), color="#4A90D9", linewidth=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform (time domain)", fontsize=13, fontweight="bold")
    ax.set_facecolor("#F7F9FC")
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "00_waveform.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✅  Saved: {path}")


def plot_spectrogram(magnitude: torch.Tensor, sr: int):
    fig, ax = plt.subplots(figsize=(10, 4))
    mag_db = 20 * np.log10(magnitude.numpy() + 1e-9)

    # Frequency axis in Hz, time axis in seconds
    n_frames = magnitude.shape[1]
    n_bins   = magnitude.shape[0]
    t_axis = np.linspace(0, n_frames * HOP_LENGTH / sr, n_frames)
    f_axis = np.linspace(0, sr / 2, n_bins)

    im = ax.pcolormesh(t_axis, f_axis, mag_db,
                       cmap="magma", shading="auto", vmin=-80, vmax=0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram — magnitude (dB)", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "00_spectrogram.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✅  Saved: {path}")


def plot_hf_removal(mag_orig: torch.Tensor, mag_removed: torch.Tensor, sr: int):
    """Plot mean magnitude spectrum before vs after HF removal."""
    fig, ax = plt.subplots(figsize=(10, 4))
    freqs = np.linspace(0, sr / 2, mag_orig.shape[0])
    mean_orig    = 20 * np.log10(mag_orig.mean(dim=1).numpy() + 1e-9)
    mean_removed = 20 * np.log10(mag_removed.mean(dim=1).numpy() + 1e-9)

    ax.plot(freqs, mean_orig, label="Original", color="#4A90D9", linewidth=1.5)
    ax.plot(freqs, mean_removed, label=f"HF removed (>{HF_CUTOFF_HZ} Hz)",
            color="#E74C3C", linewidth=1.5, linestyle="--")
    ax.axvline(HF_CUTOFF_HZ, color="#F39C12", linestyle=":", linewidth=1.2,
               label=f"Cutoff = {HF_CUTOFF_HZ} Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean magnitude (dB)")
    ax.set_title(
        f"High-frequency removal above {HF_CUTOFF_HZ} Hz\n"
        "— simulates what a person with high-freq hearing loss hears",
        fontsize=12, fontweight="bold"
    )
    ax.legend()
    ax.set_facecolor("#F7F9FC")
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "00_hf_removal_spectrum.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✅  Saved: {path}")


# ─── HF removal + ISTFT ────────────────────────────────────────────────────────
def remove_high_frequencies(waveform: torch.Tensor, sr: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Zero out STFT bins above HF_CUTOFF_HZ and invert back to audio.
    Returns: (mag_original, mag_filtered, waveform_filtered)
    """
    # Forward STFT
    window = torch.hann_window(WIN_LENGTH)
    complex_spec = torch.stft(
        waveform.squeeze(0),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        return_complex=True,
    )   # (F, T) complex

    mag_orig = complex_spec.abs()

    # Zero out bins above cutoff
    cutoff_bin = hz_to_bin(HF_CUTOFF_HZ, sr, N_FFT)
    filtered = complex_spec.clone()
    filtered[cutoff_bin:, :] = 0

    mag_filtered = filtered.abs()

    # Inverse STFT
    waveform_out = torch.istft(
        filtered,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        length=waveform.shape[-1],
    ).unsqueeze(0)

    return mag_orig, mag_filtered, waveform_out


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 0 — STFT Visualisation")
    parser.add_argument("--wav", type=str, default=None,
                        help="Path to a .wav file (optional; uses synthetic signal if omitted)")
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  PHASE 0 — STFT VISUALISATION")
    print("="*55)

    # 1. Load
    waveform, sr = load_audio(args.wav)
    duration = waveform.shape[-1] / sr
    print(f"  Duration: {duration:.2f}s  |  Samples: {waveform.shape[-1]}")

    # 2. Save original audio
    orig_path = os.path.join(DEMO_DIR, "00_original.wav")
    sf.write(orig_path, waveform.squeeze().numpy(), sr)
    print(f"  ✅  Saved: {orig_path}")

    # 3. Plot waveform
    print("\n[1/4] Plotting waveform …")
    plot_waveform(waveform, sr)

    # 4. Compute STFT + plot spectrogram
    print("[2/4] Computing STFT and plotting spectrogram …")
    magnitude, _ = compute_stft(waveform)
    plot_spectrogram(magnitude, sr)

    # 5. HF removal demo
    print(f"[3/4] Removing frequencies above {HF_CUTOFF_HZ} Hz …")
    mag_orig, mag_filtered, waveform_filtered = remove_high_frequencies(waveform, sr)
    plot_hf_removal(mag_orig, mag_filtered, sr)

    # 6. Save filtered audio
    hf_path = os.path.join(DEMO_DIR, "00_hf_removed.wav")
    sf.write(hf_path, waveform_filtered.squeeze().numpy(), sr)
    print(f"  ✅  Saved: {hf_path}")

    # 7. Print summary
    n_bins = N_FFT // 2 + 1
    cutoff_bin = hz_to_bin(HF_CUTOFF_HZ, sr, N_FFT)
    print("\n[4/4] Summary")
    print(f"  STFT parameters  →  n_fft={N_FFT}, hop={HOP_LENGTH}, win={WIN_LENGTH}")
    print(f"  Frequency resolution  →  {sr / N_FFT:.1f} Hz per bin")
    print(f"  Time resolution       →  {HOP_LENGTH / sr * 1000:.1f} ms per frame")
    print(f"  Total freq bins       →  {n_bins}  (0 Hz – {sr//2} Hz)")
    print(f"  HF cutoff bin         →  {cutoff_bin} / {n_bins}  ({HF_CUTOFF_HZ} Hz)")
    print(f"\n  🔊  Listen to the two .wav files in results/audio_demos/ to hear")
    print(f"      the effect of high-frequency loss — this is what models aim to restore.")
    print("\n" + "="*55 + "\n")


if __name__ == "__main__":
    main()
