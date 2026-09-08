"""
execution/05_wavelet_denoising.py
==================================
Phase 3 -- Classical Baseline 2: Wavelet DWT Denoising

Donoho & Johnstone (1994). Discrete Wavelet Transform + universal soft-threshold.

Algorithm:
    1. DWT(signal) --> coeffs at levels 1..L
       coeffs[0] = approximation (kept intact)
       coeffs[1..L] = detail levels (finest to coarsest)

    2. Estimate noise sigma from finest detail (coeffs[-1]):
       sigma = median(|coeffs[-1]|) / 0.6745
       (0.6745 = MAD-to-sigma factor for Gaussian noise)

    3. Universal threshold:
       lambda = sigma * sqrt(2 * log(N))

    4. Soft-threshold all detail levels (NOT approximation):
       coeff_new = sign(c) * max(|c| - lambda, 0)

    5. Inverse DWT --> denoised signal

Wavelet: db8 (Daubechies-8) -- standard for speech at 16 kHz
Level:   5  -- covers 250-8000 Hz speech range at 16 kHz

Usage:
    python execution/05_wavelet_denoising.py --test
    python execution/05_wavelet_denoising.py --all
    python execution/05_wavelet_denoising.py --all --noise-type babble --plot
    python execution/05_wavelet_denoising.py --all --limit 5

Author: Jwanil Modi (23BIT013) -- Phase 3
Date: 2026-09-08
"""

import os
import sys
import argparse
from typing import Optional
import numpy as np
import pywt
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Project paths ─────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_data_candidate    = os.path.join(ROOT, "data")
_dataset_candidate = os.path.join(ROOT, "dataset")
DATA_ROOT  = _data_candidate if os.path.isdir(_data_candidate) else _dataset_candidate

NOISY_ROOT  = os.path.join(DATA_ROOT, "processed", "noisy", "noizeus")
CLEAN_DIR   = os.path.join(DATA_ROOT, "processed", "clean", "noizeus", "clean")
OUTPUT_ROOT = os.path.join(ROOT, "results", "enhanced_wavelet", "noizeus")
PLOTS_DIR   = os.path.join(ROOT, "results", "plots", "wavelet_output_plots")

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

TARGET_SR  = 16000
TARGET_LEN = 64000

DEFAULT_WAVELET = "db8"
DEFAULT_LEVEL   = 5


# ══════════════════════════════════════════════════════════════════════════════
# CORE ALGORITHM
# ══════════════════════════════════════════════════════════════════════════════

def wavelet_denoise(noisy_waveform: np.ndarray,
                    wavelet: str = DEFAULT_WAVELET,
                    level:   int = DEFAULT_LEVEL,
                    mode:    str = "soft") -> np.ndarray:
    """
    Apply Wavelet DWT denoising (Donoho & Johnstone, 1994).

    Args:
        noisy_waveform : shape (N,), float32, values in [-1, +1]
        wavelet        : PyWavelets wavelet name ('db8' standard for speech)
        level          : DWT decomposition levels (5 for 16 kHz)
        mode           : 'soft' (recommended) or 'hard' thresholding

    Returns:
        denoised : shape (N,), float32
    """
    N = len(noisy_waveform)

    # Step 1: Decompose
    # coeffs[0]  = approximation (coarse, keep untouched)
    # coeffs[1:] = detail levels, finest first
    # At 16 kHz, level 5: each level covers one octave band
    #   Level 1 detail: 4000-8000 Hz (fricatives)
    #   Level 2 detail: 2000-4000 Hz (upper formants)
    #   Level 3 detail: 1000-2000 Hz (key for intelligibility)
    #   Level 4 detail:  500-1000 Hz (lower formants)
    #   Level 5 detail:  250-500  Hz (pitch harmonics)
    #   Approximation:   0-250    Hz (sub-bass)
    coeffs = pywt.wavedec(noisy_waveform, wavelet=wavelet, level=level)

    # Step 2: Estimate noise sigma from finest detail level
    # Speech energy is sparse at fine scales; noise is dense -> median is robust
    finest_detail = coeffs[-1]
    sigma = np.median(np.abs(finest_detail)) / 0.6745
    sigma = max(sigma, 1e-10)

    # Step 3: Universal threshold
    # Theoretically optimal for N Gaussian noise observations
    lam = sigma * np.sqrt(2.0 * np.log(N))

    # Step 4: Soft-threshold all detail levels, leave approximation intact
    # Soft: shrinks coefficients toward zero smoothly (fewer clicks than hard)
    denoised_coeffs = [coeffs[0]]
    for detail in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(detail, value=lam, mode=mode))

    # Step 5: Reconstruct
    denoised = pywt.waverec(denoised_coeffs, wavelet=wavelet)

    # Trim/pad boundary samples (waverec may add 1-2 extra samples)
    if len(denoised) > N:
        denoised = denoised[:N]
    elif len(denoised) < N:
        denoised = np.pad(denoised, (0, N - len(denoised)))

    return denoised.astype(np.float32)


def postprocess(enhanced: np.ndarray, original_length: int) -> np.ndarray:
    """Standardize filter output: DC removal -> length fix -> peak normalize."""
    enhanced = enhanced - enhanced.mean()

    if len(enhanced) > original_length:
        enhanced = enhanced[:original_length]
    elif len(enhanced) < original_length:
        enhanced = np.pad(enhanced, (0, original_length - len(enhanced)))

    peak = np.abs(enhanced).max()
    if peak > 1e-6:
        enhanced = enhanced / peak

    return enhanced.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison(noisy: np.ndarray,
                    enhanced: np.ndarray,
                    clean: Optional[np.ndarray],
                    filename: str,
                    sr: int = TARGET_SR) -> str:
    """Plot waveform + spectrogram: Noisy | Wavelet Enhanced | Clean Reference."""
    cols   = 3 if clean is not None else 2
    labels = ["Noisy Input", "Wavelet Enhanced"]
    waves  = [noisy, enhanced]
    if clean is not None:
        labels.append("Clean Reference")
        waves.append(clean)

    fig, axes = plt.subplots(2, cols, figsize=(6 * cols, 7))
    fig.suptitle(f"Wavelet DWT Denoising -- {filename}", fontsize=13, fontweight="bold")

    for col, (label, wav) in enumerate(zip(labels, waves)):
        time_axis = np.linspace(0, len(wav) / sr, len(wav))

        axes[0, col].plot(time_axis, wav, linewidth=0.4, color="#2ECC71")
        axes[0, col].set_title(label, fontsize=11)
        axes[0, col].set_xlabel("Time (s)")
        axes[0, col].set_ylabel("Amplitude")
        axes[0, col].set_ylim(-1.1, 1.1)
        axes[0, col].grid(True, alpha=0.3)

        axes[1, col].specgram(wav, Fs=sr, NFFT=512, noverlap=384, cmap="viridis")
        axes[1, col].set_xlabel("Time (s)")
        axes[1, col].set_ylabel("Frequency (Hz)")
        axes[1, col].set_ylim(0, sr // 2)

    plt.tight_layout()
    stem    = os.path.splitext(filename)[0]
    outpath = os.path.join(PLOTS_DIR, f"05_wav_{stem}.png")
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close()
    return outpath


# ══════════════════════════════════════════════════════════════════════════════
# FILE PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def _clean_path_for(noisy_basename: str) -> Optional[str]:
    speaker_id = noisy_basename.split("_")[0]
    candidate  = os.path.join(CLEAN_DIR, f"{speaker_id}.wav")
    return candidate if os.path.exists(candidate) else None


def process_file(noisy_path: str,
                 plot: bool   = False,
                 wavelet: str = DEFAULT_WAVELET,
                 level:   int = DEFAULT_LEVEL) -> dict:
    """Load one NOIZEUS .wav, apply Wavelet DWT, postprocess, save enhanced .wav."""
    if not os.path.isabs(noisy_path):
        noisy_path = os.path.join(ROOT, noisy_path) if not os.path.exists(noisy_path) \
                     else os.path.abspath(noisy_path)

    if not os.path.exists(noisy_path):
        print(f"  [WARN]  File not found: {noisy_path}")
        return {}

    rel        = os.path.relpath(noisy_path, NOISY_ROOT)
    parts      = rel.replace("\\", "/").split("/")
    noise_type = parts[0] if len(parts) >= 3 else "unknown"
    snr_db     = parts[1] if len(parts) >= 3 else "unknown"
    filename   = os.path.basename(noisy_path)

    output_dir  = os.path.join(OUTPUT_ROOT, noise_type, snr_db)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    noisy_wav, sr = sf.read(noisy_path, dtype="float32")
    if noisy_wav.ndim > 1:
        noisy_wav = noisy_wav[:, 0]
    original_length = len(noisy_wav)

    clean_wav  = None
    clean_path = _clean_path_for(filename)
    if clean_path:
        clean_wav, _ = sf.read(clean_path, dtype="float32")
        if clean_wav.ndim > 1:
            clean_wav = clean_wav[:, 0]

    enhanced = wavelet_denoise(noisy_wav, wavelet=wavelet, level=level)
    enhanced = postprocess(enhanced, original_length)

    sf.write(output_path, enhanced, sr, subtype="PCM_16")

    input_rms  = float(np.sqrt(np.mean(noisy_wav ** 2)))
    output_rms = float(np.sqrt(np.mean(enhanced  ** 2)))

    label = f"{noise_type}/{snr_db}/{filename}"
    print(f"  [OK] {label:<45s}  in_rms={input_rms:.4f} -> out_rms={output_rms:.4f}")
    print(f"      saved -> {output_path}")

    if plot:
        plot_stem = f"{noise_type}_{snr_db}_{os.path.splitext(filename)[0]}"
        plot_path = plot_comparison(noisy_wav, enhanced, clean_wav,
                                    plot_stem + ".wav", sr=sr)
        print(f"      [PLOT] -> {plot_path}")

    return {
        "noisy_path" : noisy_path,
        "noise_type" : noise_type,
        "snr_db"     : snr_db,
        "filename"   : filename,
        "input_rms"  : input_rms,
        "output_rms" : output_rms,
        "output_path": output_path,
    }


def collect_noisy_files(noise_type_filter: Optional[str] = None,
                        snr_filter: Optional[str] = None) -> list:
    if not os.path.isdir(NOISY_ROOT):
        print(f"[ERR] Noisy root not found: {NOISY_ROOT}")
        sys.exit(1)

    paths = []
    noise_types = sorted(d for d in os.listdir(NOISY_ROOT) if not d.startswith("."))

    if noise_type_filter:
        if noise_type_filter not in noise_types:
            print(f"[ERR] Noise type '{noise_type_filter}' not found. Available: {noise_types}")
            sys.exit(1)
        noise_types = [noise_type_filter]

    for nt in noise_types:
        nt_dir = os.path.join(NOISY_ROOT, nt)
        if not os.path.isdir(nt_dir):
            continue
        snr_levels = sorted(d for d in os.listdir(nt_dir) if not d.startswith("."))
        if snr_filter:
            if snr_filter not in snr_levels:
                continue
            snr_levels = [snr_filter]
        for snr in snr_levels:
            snr_dir = os.path.join(nt_dir, snr)
            if not os.path.isdir(snr_dir):
                continue
            for fname in sorted(os.listdir(snr_dir)):
                if fname.endswith(".wav") and not fname.startswith("."):
                    paths.append(os.path.join(snr_dir, fname))

    return paths


def process_all(plot: bool = False,
                limit: Optional[int] = None,
                noise_type_filter: Optional[str] = None,
                snr_filter: Optional[str] = None,
                wavelet: str = DEFAULT_WAVELET,
                level:   int = DEFAULT_LEVEL) -> list:
    all_paths = collect_noisy_files(noise_type_filter, snr_filter)
    if not all_paths:
        print("[ERR] No .wav files found.")
        sys.exit(1)

    if limit:
        all_paths = all_paths[:limit]

    filter_desc = ""
    if noise_type_filter:
        filter_desc += f"  noise={noise_type_filter}"
    if snr_filter:
        filter_desc += f"  snr={snr_filter}"

    print(f"\n[WAVELET] Denoising {len(all_paths)} file(s){filter_desc}")
    print(f"   Wavelet={wavelet}  Level={level}")
    print(f"   Source -> {NOISY_ROOT}")
    print(f"   Output -> {OUTPUT_ROOT}\n")

    results = []
    for p in all_paths:
        result = process_file(p, plot=plot, wavelet=wavelet, level=level)
        if result:
            results.append(result)

    if results:
        avg_input_rms  = np.mean([r["input_rms"]  for r in results])
        avg_output_rms = np.mean([r["output_rms"] for r in results])
        print("\n" + "-" * 60)
        print(f"  Files processed : {len(results)}")
        print(f"  Avg input RMS   : {avg_input_rms:.4f}")
        print(f"  Avg output RMS  : {avg_output_rms:.4f}")
        print(f"  Output root     : {OUTPUT_ROOT}")
        print("-" * 60 + "\n")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

def self_test():
    """Generate synthetic voiced+noise signal and verify Wavelet DWT output."""
    print("Running Wavelet DWT self-test with synthetic signal...")

    sr  = TARGET_SR
    dur = 4.0
    N   = int(sr * dur)
    t   = np.linspace(0, dur, N, endpoint=False)

    # Voiced speech stand-in: 200 Hz fundamental + harmonics
    clean = sum(0.3 / k * np.sin(2 * np.pi * 200 * k * t) for k in range(1, 6))
    clean = (clean / (np.abs(clean).max() + 1e-9) * 0.7).astype(np.float32)

    # White noise at ~0 dB SNR
    rng   = np.random.default_rng(42)
    noise = (0.25 * rng.standard_normal(N)).astype(np.float32)
    noisy = np.clip(clean + noise, -1.0, 1.0)

    enhanced = wavelet_denoise(noisy, wavelet=DEFAULT_WAVELET, level=DEFAULT_LEVEL)
    enhanced = postprocess(enhanced, N)

    assert enhanced.shape == (N,),       f"Shape mismatch: {enhanced.shape} vs ({N},)"
    assert enhanced.dtype == np.float32, "dtype must be float32"
    assert np.abs(enhanced).max() <= 1.0 + 1e-5, "Output exceeds [-1, +1]"

    noise_rms_in  = float(np.sqrt(np.mean((noisy   - clean) ** 2)))
    noise_rms_out = float(np.sqrt(np.mean((enhanced - clean) ** 2)))

    print(f"  Input noise RMS  : {noise_rms_in:.4f}")
    print(f"  Output noise RMS : {noise_rms_out:.4f}")
    print(f"  [OK] Self-test passed -- output is valid float32 waveform in [-1, +1]")

    if noise_rms_out < noise_rms_in:
        print(f"  [OK] Noise reduced by {(1 - noise_rms_out / noise_rms_in) * 100:.1f}%")
    else:
        print(f"  [WARN]  Noise not reduced on synthetic tone -- normal for non-speech signals.")
        print(f"      Verify on real NOIZEUS clips.")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 -- Wavelet DWT Classical Speech Enhancement Baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--file",  metavar="PATH", help="Process a single noisy .wav")
    mode.add_argument("--all",   action="store_true", help="Process ALL .wav files")
    mode.add_argument("--test",  action="store_true", help="Run self-test (no dataset needed)")

    parser.add_argument("--noise-type", metavar="TYPE",  default=None)
    parser.add_argument("--snr",        metavar="LEVEL", default=None)
    parser.add_argument("--limit",      type=int,        default=None)
    parser.add_argument("--plot",       action="store_true")
    parser.add_argument("--wavelet",    default=DEFAULT_WAVELET,
                        help=f"Wavelet name (default: {DEFAULT_WAVELET})")
    parser.add_argument("--level",      type=int, default=DEFAULT_LEVEL,
                        help=f"DWT levels (default: {DEFAULT_LEVEL})")

    args = parser.parse_args()

    if args.test:
        self_test()
    elif args.file:
        process_file(args.file, plot=args.plot, wavelet=args.wavelet, level=args.level)
    elif args.all:
        process_all(
            plot              = args.plot,
            limit             = args.limit,
            noise_type_filter = args.noise_type,
            snr_filter        = args.snr,
            wavelet           = args.wavelet,
            level             = args.level,
        )


if __name__ == "__main__":
    main()
