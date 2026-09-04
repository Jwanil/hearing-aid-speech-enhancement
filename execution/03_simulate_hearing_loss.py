from __future__ import annotations
"""
Phase 1 — Audiology: Simulate Hearing Loss
===========================================
Directive: directives/01_audiology.md

What this script does:
  Uses the MSBG (Moore-Stone-Baer-Glasberg) hearing loss model from pyclarity
  to simulate what speech sounds like to a person with a given audiogram.

  MSBG models the full auditory pathway:
    1. Outer/middle ear transfer function
    2. Cochlear filtering (auditory filter bank)
    3. Outer hair cell (OHC) damage → reduced compression
    4. Inner hair cell (IHC) damage → elevated threshold
  This is the same simulator used in the Clarity Challenge (CEC2/CEC3).

  The script:
    1. Loads a .wav file (or uses the synthetic signal from Phase 0)
    2. Applies all 3 test audiogram profiles
    3. Saves 3 output .wav files (one per profile)
    4. Plots a 4-panel spectrogram comparison (original + 3 profiles)

  IMPORTANT: MSBG requires 44100 Hz sample rate (resampled internally if needed).

Usage:
    python execution/03_simulate_hearing_loss.py
    python execution/03_simulate_hearing_loss.py --wav path/to/speech.wav

Outputs:
    results/audio_demos/01_original_44k.wav
    results/audio_demos/01_mild_flat_loss.wav
    results/audio_demos/01_moderate_sloping_loss.wav
    results/audio_demos/01_severe_hf_loss.wav
    results/plots/01_hearing_loss_spectrograms.png
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T

warnings.filterwarnings("ignore")

# pyclarity imports
from clarity.evaluator.msbg.msbg import Ear
from clarity.utils.audiogram import Audiogram

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR = os.path.join(ROOT, "results", "audio_demos")
PLOT_DIR = os.path.join(ROOT, "results", "plots")
os.makedirs(DEMO_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ─── MSBG requires 44100 Hz ───────────────────────────────────────────────────
MSBG_SR = 44100

# ─── The 3 project evaluation audiogram profiles ─────────────────────────────
FREQUENCIES = np.array([250, 500, 1000, 2000, 4000, 8000], dtype=float)

TEST_PROFILES = {
    "Mild Flat Loss":
        np.array([15, 20, 25, 25, 30, 30], dtype=float),
    "Moderate Sloping (Presbycusis)":
        np.array([10, 15, 30, 50, 65, 80], dtype=float),
    "Severe HF Loss":
        np.array([10, 10, 35, 65, 85, 90], dtype=float),
}

PROFILE_COLORS = ["#3498DB", "#E74C3C", "#2ECC71"]


# ─── Audio loading ────────────────────────────────────────────────────────────
def load_audio_44k(wav_path: str | None) -> np.ndarray:
    """
    Load a .wav file and resample to 44100 Hz mono float64 (required by MSBG).
    If no file supplied, use Phase 0's synthetic audio or generate a fresh tone.
    """
    fallback = os.path.join(DEMO_DIR, "00_original.wav")

    if wav_path and os.path.exists(wav_path):
        source = wav_path
    elif os.path.exists(fallback):
        print(f"  No --wav supplied. Using Phase 0 demo: {fallback}")
        source = fallback
    else:
        # Generate a 2-second speech-like signal
        print("  No audio found. Generating synthetic signal …")
        sr_tmp = 44100
        t = np.linspace(0, 2.0, sr_tmp * 2)
        # Voiced part
        sig = sum(0.3 / k * np.sin(2 * np.pi * 100 * k * t) for k in range(1, 9))
        sig = sig / (np.abs(sig).max() + 1e-9) * 0.5
        tmp_path = os.path.join(DEMO_DIR, "_tmp_synth.wav")
        sf.write(tmp_path, sig, sr_tmp)
        source = tmp_path

    waveform, sr = torchaudio.load(source)

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 44100 if needed
    if sr != MSBG_SR:
        resampler = T.Resample(sr, MSBG_SR)
        waveform = resampler(waveform)
        print(f"  Resampled {sr} Hz → {MSBG_SR} Hz")

    # MSBG expects float64 numpy array, shape (N,)
    signal_np = waveform.squeeze().numpy().astype(np.float64)

    # Normalise to avoid clipping in MSBG
    peak = np.abs(signal_np).max()
    if peak > 0:
        signal_np = signal_np / peak * 0.5

    print(f"  Audio loaded: {len(signal_np)/MSBG_SR:.2f}s  |  SR={MSBG_SR} Hz")
    return signal_np


# ─── Hearing loss simulation ──────────────────────────────────────────────────
def simulate_hearing_loss(signal: np.ndarray,
                           levels: np.ndarray,
                           label: str) -> np.ndarray:
    """
    Apply MSBG hearing loss simulation for the given audiogram levels.
    Returns the processed signal (same length as input, float64).
    """
    audiogram = Audiogram(levels=levels, frequencies=FREQUENCIES)
    ear = Ear(sample_rate=float(MSBG_SR))
    ear.set_audiogram(audiogram)

    result = ear.process(signal)    # returns list with one array per channel
    processed = result[0].astype(np.float64)

    # Trim or pad to match input length (MSBG may add a few samples)
    if len(processed) > len(signal):
        processed = processed[:len(signal)]
    elif len(processed) < len(signal):
        processed = np.pad(processed, (0, len(signal) - len(processed)))

    # Normalise output
    peak = np.abs(processed).max()
    if peak > 0:
        processed = processed / peak * 0.45

    return processed


# ─── Spectrogram comparison plot ─────────────────────────────────────────────
def plot_comparison(original: np.ndarray,
                    processed: dict[str, np.ndarray]):
    n_panels = 1 + len(processed)
    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 5, 4), sharey=True)

    all_signals = {"Original (no loss)": original}
    all_signals.update(processed)
    colors = ["#7F8C8D"] + PROFILE_COLORS

    for ax, (title, sig), color in zip(axes, all_signals.items(), colors):
        # Compute STFT magnitude
        n_fft = 1024
        hop   = 256
        win   = torch.hann_window(n_fft)
        t_sig = torch.tensor(sig, dtype=torch.float32)
        spec  = torch.stft(t_sig, n_fft=n_fft, hop_length=hop,
                           win_length=n_fft, window=win, return_complex=True)
        mag_db = 20 * np.log10(spec.abs().numpy() + 1e-9)

        n_frames = mag_db.shape[1]
        n_bins   = mag_db.shape[0]
        t_ax = np.linspace(0, len(sig) / MSBG_SR, n_frames)
        f_ax = np.linspace(0, MSBG_SR / 2, n_bins)

        im = ax.pcolormesh(t_ax, f_ax / 1000, mag_db,   # f in kHz
                           cmap="magma", shading="auto", vmin=-80, vmax=0)
        ax.set_title(title, fontsize=9, fontweight="bold", color=color if color != "#7F8C8D" else "black")
        ax.set_xlabel("Time (s)", fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel("Frequency (kHz)", fontsize=9)

    fig.colorbar(im, ax=axes[-1], label="dB", fraction=0.046, pad=0.04)
    fig.suptitle("Hearing Loss Simulation — Spectrogram Comparison\n"
                 "(Energy loss at high frequencies visible in right panels)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "01_hearing_loss_spectrograms.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✅  Saved: {path}")


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 1 — Hearing Loss Simulation")
    parser.add_argument("--wav", type=str, default=None,
                        help="Path to a .wav file (optional)")
    args = parser.parse_args()

    print("\n" + "="*57)
    print("  PHASE 1 — HEARING LOSS SIMULATION (MSBG)")
    print("="*57)
    print("\n  The MSBG model simulates the full auditory pathway:")
    print("  outer ear → cochlear filtering → OHC/IHC damage")

    # 1. Load audio
    print("\n[1/3] Loading audio …")
    signal = load_audio_44k(args.wav)

    # 2. Save original at 44k
    orig_path = os.path.join(DEMO_DIR, "01_original_44k.wav")
    sf.write(orig_path, signal, MSBG_SR)
    print(f"  ✅  Saved: {orig_path}")

    # 3. Simulate each profile
    print("\n[2/3] Applying hearing loss simulation …")
    processed = {}
    safe_names = {
        "Mild Flat Loss":                   "01_mild_flat_loss",
        "Moderate Sloping (Presbycusis)":   "01_moderate_sloping_loss",
        "Severe HF Loss":                   "01_severe_hf_loss",
    }

    for name, levels in TEST_PROFILES.items():
        print(f"\n  Profile: {name}")
        print(f"  Levels (dB HL): {levels.tolist()} @ {FREQUENCIES.astype(int).tolist()} Hz")
        out = simulate_hearing_loss(signal, levels, name)
        processed[name] = out

        fname = safe_names[name] + ".wav"
        out_path = os.path.join(DEMO_DIR, fname)
        sf.write(out_path, out, MSBG_SR)
        print(f"  ✅  Saved: {out_path}")

    # 4. Plot comparison
    print("\n[3/3] Plotting spectrogram comparison …")
    plot_comparison(signal, processed)

    # 5. Summary
    print("\n[Summary]")
    print("  Listen to the .wav files in results/audio_demos/:")
    print("    01_original_44k.wav          ← clean, unprocessed")
    print("    01_mild_flat_loss.wav         ← mild, roughly uniform loss")
    print("    01_moderate_sloping_loss.wav  ← presbycusis: s/sh/f sounds dulled")
    print("    01_severe_hf_loss.wav         ← speech becomes muffled, consonants lost")
    print()
    print("  ℹ️  What to listen for:")
    print("     - Fricatives ('s', 'sh', 'f', 'th') live at 4–8 kHz")
    print("     - Sloping loss removes them progressively")
    print("     - This is EXACTLY what your model must restore using the audiogram vector")
    print()
    print("  ℹ️  The audiogram vector [10, 15, 30, 50, 65, 80] becomes the")
    print("     FiLM conditioning input to your U-Net and Mamba models.")
    print("\n" + "="*57 + "\n")


if __name__ == "__main__":
    main()
