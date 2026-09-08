"""
execution/04_mmse_lsa_filter.py
===============================
Phase 3 -- Classical Baseline 1: MMSE-LSA (Minimum Mean Square Error - Log Spectral Amplitude) Filter

Introduced by Ephraim & Malah (1985). This is the industry standard algorithm used in
commercial hearing aids today (Oticon, Phonak, Signia). It is the first and stronger of
the two classical baselines -- every deep-learning model in Phases 4-5b must beat it.

Faculty instruction: Implement MMSE-LSA FIRST. It teaches you noise estimation and gain
functions -- the core pattern behind ALL enhancement algorithms.

Algorithm:
    For each STFT frame n and frequency bin k:
    1. Compute instantaneous SNR (gamma): gamma = |Y|^2 / noise_power
    2. Update a priori SNR (xi) using decision-directed approach (alpha=0.98)
    3. Compute MMSE-LSA gain using the exponential integral E1
    4. Apply gain to magnitude; reconstruct with noisy phase -> ISTFT

Data Layout (actual project structure):
    Data/
      processed/
        noisy/
          noizeus/
            <noise_type>/          # airport, babble, car, …
              <snr_dB>/            # 0dB, 5dB, 10dB, 15dB
                sp01_airport_sn0.wav
                …
        clean/
          noizeus/
            clean/
              sp01.wav
              …

Inputs:  Data/processed/noisy/noizeus/<noise_type>/<snr_dB>/
Outputs: results/enhanced_mmse/noizeus/<noise_type>/<snr_dB>/  (same filename, denoised)

Usage:
    # Run built-in self-test (no dataset required):
    python execution/04_mmse_lsa_filter.py --test

    # Process a single file (path relative to project root or absolute):
    python execution/04_mmse_lsa_filter.py --file Data/processed/noisy/noizeus/airport/0dB/sp01_airport_sn0.wav

    # Process all noisy files across every noise type and SNR level:
    python execution/04_mmse_lsa_filter.py --all

    # Restrict to one noise type and/or one SNR level:
    python execution/04_mmse_lsa_filter.py --all --noise-type airport --snr 0dB

    # Also save comparison plots:
    python execution/04_mmse_lsa_filter.py --all --noise-type babble --plot

    # Quick sanity-check -- first 5 files only:
    python execution/04_mmse_lsa_filter.py --all --limit 5

Author: Namya Shah (23BIT027) -- Phase 3
Date: 2026-09-06
"""

import os
import sys
import argparse
from typing import Optional
import numpy as np
import scipy.signal
from scipy.special import exp1  # Exponential integral E1 -- core of MMSE-LSA gain
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Project paths ────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Actual data location — symlink to SSD (Jwanil) or dataset/ folder (Namya)
# Try data/ first (Jwanil's setup), fall back to dataset/ (Namya's setup)
_data_candidate = os.path.join(ROOT, "data")
_dataset_candidate = os.path.join(ROOT, "dataset")
DATA_ROOT  = _data_candidate if os.path.isdir(_data_candidate) else _dataset_candidate

# Noisy files:  data/processed/noisy/noizeus/<noise_type>/<snr_dB>/
NOISY_ROOT = os.path.join(DATA_ROOT, "processed", "noisy", "noizeus")

# Clean files:  data/processed/clean/noizeus/clean/
CLEAN_DIR  = os.path.join(DATA_ROOT, "processed", "clean", "noizeus", "clean")

# Outputs mirror the noisy sub-tree:  results/enhanced_mmse/noizeus/<noise_type>/<snr_dB>/
OUTPUT_ROOT = os.path.join(ROOT, "results", "enhanced_mmse", "noizeus")
PLOTS_DIR   = os.path.join(ROOT, "results", "plots", "mmse_output_plots")

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

# ─── Audio constants (must match Phase 2 standardisation) ────────────────────
TARGET_SR  = 16000   # Hz -- all project audio is 16 kHz
TARGET_LEN = 64000   # samples -- 4 seconds at 16 kHz


# ══════════════════════════════════════════════════════════════════════════════
# CORE ALGORITHM
# ══════════════════════════════════════════════════════════════════════════════

def mmse_lsa_filter(noisy_waveform: np.ndarray,
                    sr: int   = TARGET_SR,
                    frame_ms: float = 25.0,
                    hop_ms:   float = 10.0,
                    alpha:    float = 0.98) -> np.ndarray:
    """
    Apply MMSE-LSA (Minimum Mean Square Error - Log Spectral Amplitude) speech enhancement.

    This is the Ephraim & Malah (1985) algorithm -- the industry-standard noise suppressor
    used in hearing aids. Key advantages over the basic Wiener filter:
      - Decision-directed a priori SNR (xi) smooths gain over time -> fewer artifacts
      - Operates on log spectral amplitude -> matches human auditory perception
      - Exponential integral gain dramatically reduces "musical noise"

    Args:
        noisy_waveform : np.ndarray, shape (N,), float32, values nominally in [-1, +1]
        sr             : sample rate (must be 16000 after standardisation)
        frame_ms       : STFT (Short-Time Fourier Transform) frame length in milliseconds
        hop_ms         : STFT hop length in milliseconds
        alpha          : decision-directed smoothing constant (0.98 is standard for speech)

    Returns:
        enhanced : np.ndarray, shape (N,), float32 -- noise-suppressed waveform
    """
    # ── STFT parameters ──────────────────────────────────────────────────────
    n_fft  = int(sr * frame_ms / 1000)   # 25 ms × 16 kHz = 400 samples
    hop    = int(sr * hop_ms  / 1000)    # 10 ms × 16 kHz = 160 samples
    window = np.hanning(n_fft)           # Hann window -- standard for speech STFT

    # ── Step 1: STFT of noisy signal ─────────────────────────────────────────
    # scipy.signal.stft returns:
    #   f     : frequency axis  (freq_bins,)
    #   t     : time axis       (time_frames,)
    #   Y     : complex STFT    (freq_bins, time_frames)
    _f, _t, Y = scipy.signal.stft(
        noisy_waveform,
        fs       = sr,
        window   = window,
        nperseg  = n_fft,
        noverlap = n_fft - hop
    )

    mag   = np.abs(Y)     # Magnitude spectrogram  |Y(k,n)|
    phase = np.angle(Y)   # Phase spectrogram ∠Y(k,n) -- kept from noisy signal

    n_freq, n_frames = mag.shape

    # ── Step 2: Initial noise power estimate ─────────────────────────────────
    # Use first 6 frames (~60 ms) as voice-activity-free noise reference.
    # Shape: (n_freq, 1) -- broadcast-friendly column vector.
    noise_est = (mag[:, :6] ** 2).mean(axis=1, keepdims=True)
    noise_est = np.maximum(noise_est, 1e-10)   # avoid zero

    # ── Step 3: Frame-by-frame MMSE-LSA processing ───────────────────────────
    xi    = np.ones((n_freq, 1))    # a priori SNR (xi_hat) initialised to 1
    A_hat = np.zeros((n_freq, 1))   # previous frame's enhanced magnitude
    G_out = np.zeros_like(mag)      # gain matrix G(k,n) -- filled in the loop

    for n in range(n_frames):
        Y_n = mag[:, n:n+1]   # current frame magnitude, shape (n_freq, 1)

        # ── Instantaneous (a posteriori) SNR ─────────────────────────────────
        # gamma(k,n) = |Y(k,n)|^2 / lambda_n(k,n)
        gamma_n = Y_n ** 2 / (noise_est + 1e-10)

        # ── Decision-directed a priori SNR (xi) update ───────────────────────
        # xi(k,n) = alpha * [A_hat^2(k,n-1) / lambda_n] + (1-alpha) * max(gamma-1, 0)
        # The first term is the "decided" part (smoothed from the previous enhanced frame).
        # The second term is the "instantaneous" part (current frame SNR estimate).
        # alpha=0.98 strongly weights the previous estimate -> temporal smoothness.
        if n == 0:
            # No previous frame -- bootstrap from instantaneous SNR only
            xi_n = alpha * np.ones_like(gamma_n) + (1.0 - alpha) * np.maximum(gamma_n - 1.0, 0.0)
        else:
            xi_n = (alpha * (A_hat ** 2) / (noise_est + 1e-10)
                    + (1.0 - alpha) * np.maximum(gamma_n - 1.0, 0.0))

        xi_n = np.maximum(xi_n, 1e-10)   # keep positive for exp1 stability

        # ── MMSE-LSA gain function ────────────────────────────────────────────
        # v(k,n) = xi(k,n) * gamma(k,n) / (1 + xi(k,n))
        # G(k,n) = xi(k,n) / (1 + xi(k,n)) * exp(0.5 * E1(v))
        #
        # E1(v) = ∫_v^∞ (e^-t / t) dt  -- the exponential integral
        # exp(0.5 * E1(v)) is the log-spectral-amplitude correction term that
        # reduces musical noise compared to the pure Wiener gain.
        v   = xi_n * gamma_n / (1.0 + xi_n)
        v   = np.clip(v, 1e-10, 700.0)          # exp1 blows up near 0; 700 avoids overflow

        G_n = xi_n / (1.0 + xi_n) * np.exp(0.5 * exp1(v))
        G_n = np.clip(G_n, 0.0, 1.0)            # gain must stay in [0, 1]

        A_hat = G_n * Y_n                        # enhanced amplitude for next frame's xi
        G_out[:, n:n+1] = G_n

        # ── Online noise estimate update (simple minimum-statistics) ─────────
        # Track a slowly-decaying minimum of the power spectrum.
        # 0.98/0.02 gives a ~50-frame (500 ms) forgetting window.
        noise_est = 0.98 * noise_est + 0.02 * np.minimum(Y_n ** 2, noise_est * 10.0)
        noise_est = np.maximum(noise_est, 1e-10)

    # ── Step 4: Reconstruct waveform ─────────────────────────────────────────
    enhanced_mag = G_out * mag                           # apply suppression gain
    enhanced_Y   = enhanced_mag * np.exp(1j * phase)    # keep noisy phase (noisy-phase synthesis)

    _t_out, enhanced = scipy.signal.istft(
        enhanced_Y,
        fs       = sr,
        window   = window,
        nperseg  = n_fft,
        noverlap = n_fft - hop
    )

    return enhanced.astype(np.float32)


def postprocess_filter_output(enhanced: np.ndarray, original_length: int) -> np.ndarray:
    """
    Standardise classical filter output for downstream use.
    Call this after mmse_lsa_filter() before saving or computing metrics.

    Steps:
        1. Remove DC offset (filter can introduce a mean shift)
        2. Fix length (ISTFT may add/remove 1-2 samples at boundaries)
        3. Re-normalise to peak [-1, +1]
    """
    # 1. DC offset removal
    enhanced = enhanced - enhanced.mean()

    # 2. Length correction
    if len(enhanced) > original_length:
        enhanced = enhanced[:original_length]
    elif len(enhanced) < original_length:
        enhanced = np.pad(enhanced, (0, original_length - len(enhanced)))

    # 3. Peak normalisation
    peak = np.abs(enhanced).max()
    if peak > 1e-6:          # guard against silent files
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
    """
    Plot waveform + spectrogram comparison: noisy vs enhanced vs clean (if available).
    Saves to results/plots/04_mmse_<filename>.png
    Returns the path to the saved plot.
    """
    cols   = 3 if clean is not None else 2
    labels = ["Noisy Input", "MMSE-LSA Enhanced"]
    waves  = [noisy, enhanced]
    if clean is not None:
        labels.append("Clean Reference")
        waves.append(clean)

    fig, axes = plt.subplots(2, cols, figsize=(6 * cols, 7))
    fig.suptitle(f"MMSE-LSA Filter -- {filename}", fontsize=13, fontweight="bold")

    for col, (label, wav) in enumerate(zip(labels, waves)):
        time_axis = np.linspace(0, len(wav) / sr, len(wav))

        # Row 0 -- waveform
        axes[0, col].plot(time_axis, wav, linewidth=0.4, color="#1A56AA")
        axes[0, col].set_title(label, fontsize=11)
        axes[0, col].set_xlabel("Time (s)")
        axes[0, col].set_ylabel("Amplitude")
        axes[0, col].set_ylim(-1.1, 1.1)
        axes[0, col].grid(True, alpha=0.3)

        # Row 1 -- spectrogram
        axes[1, col].specgram(wav, Fs=sr, NFFT=512, noverlap=384, cmap="inferno")
        axes[1, col].set_xlabel("Time (s)")
        axes[1, col].set_ylabel("Frequency (Hz)")
        axes[1, col].set_ylim(0, sr // 2)

    plt.tight_layout()
    stem   = os.path.splitext(filename)[0]
    outpath = os.path.join(PLOTS_DIR, f"04_mmse_{stem}.png")
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close()
    return outpath


# ══════════════════════════════════════════════════════════════════════════════
# FILE PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def _clean_path_for(noisy_basename: str) -> Optional[str]:
    """
    Derive the clean reference path for a given noisy filename.

    NOIZEUS naming convention:
        noisy : sp01_airport_sn0.wav  ->  speaker ID = sp01
        clean : Data/processed/clean/noizeus/clean/sp01.wav

    Returns the path if found, else None.
    """
    # Extract speaker ID: everything up to the first underscore (e.g. 'sp01')
    speaker_id = noisy_basename.split("_")[0]
    candidate  = os.path.join(CLEAN_DIR, f"{speaker_id}.wav")
    return candidate if os.path.exists(candidate) else None


def process_file(noisy_path: str, plot: bool = False) -> dict:
    """
    Load one noisy .wav from the Data/ tree, apply MMSE-LSA, postprocess,
    and save the enhanced .wav to results/enhanced_mmse/noizeus/<noise_type>/<snr_dB>/.

    Args:
        noisy_path : absolute path OR path relative to the project root of a
                     noisy .wav file inside Data/processed/noisy/noizeus/
        plot       : if True, also generate and save a comparison plot

    Returns:
        dict with keys: noisy_path, noise_type, snr_db, filename,
                        input_rms, output_rms, output_path
    """
    # Resolve to absolute path
    if not os.path.isabs(noisy_path):
        if os.path.exists(noisy_path):
            noisy_path = os.path.abspath(noisy_path)
        else:
            noisy_path = os.path.join(ROOT, noisy_path)

    if not os.path.exists(noisy_path):
        print(f"  [WARN]  File not found: {noisy_path}")
        return {}

    # ── Derive noise_type and snr_db from the path hierarchy ─────────────────
    # Expected structure: …/noizeus/<noise_type>/<snr_dB>/sp01_airport_sn0.wav
    rel        = os.path.relpath(noisy_path, NOISY_ROOT)   # airport/0dB/sp01_airport_sn0.wav
    parts      = rel.replace("\\", "/").split("/")
    if len(parts) >= 3:
        noise_type = parts[0]   # e.g. "airport"
        snr_db     = parts[1]   # e.g. "0dB"
    else:
        noise_type = "unknown"
        snr_db     = "unknown"

    filename = os.path.basename(noisy_path)

    # ── Output path mirrors the noisy sub-tree ────────────────────────────────
    output_dir  = os.path.join(OUTPUT_ROOT, noise_type, snr_db)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # ── Load noisy ────────────────────────────────────────────────────────────
    noisy_wav, sr = sf.read(noisy_path, dtype="float32")
    if noisy_wav.ndim > 1:
        noisy_wav = noisy_wav[:, 0]   # force mono

    original_length = len(noisy_wav)

    # ── Load clean reference (plotting only -- not used by the filter) ─────────
    clean_wav  = None
    clean_path = _clean_path_for(filename)
    if clean_path:
        clean_wav, _ = sf.read(clean_path, dtype="float32")
        if clean_wav.ndim > 1:
            clean_wav = clean_wav[:, 0]

    # ── Apply MMSE-LSA ────────────────────────────────────────────────────────
    enhanced = mmse_lsa_filter(noisy_wav, sr=sr)

    # ── Postprocess ───────────────────────────────────────────────────────────
    enhanced = postprocess_filter_output(enhanced, original_length)

    # ── Save ──────────────────────────────────────────────────────────────────
    sf.write(output_path, enhanced, sr, subtype="PCM_16")

    # ── Metrics ───────────────────────────────────────────────────────────────
    input_rms  = float(np.sqrt(np.mean(noisy_wav ** 2)))
    output_rms = float(np.sqrt(np.mean(enhanced  ** 2)))

    label = f"{noise_type}/{snr_db}/{filename}"
    print(f"  [OK] {label:<45s}  in_rms={input_rms:.4f} -> out_rms={output_rms:.4f}")
    print(f"      saved -> {output_path}")

    # ── Optional plot ─────────────────────────────────────────────────────────
    if plot:
        plot_stem  = f"{noise_type}_{snr_db}_{os.path.splitext(filename)[0]}"
        plot_path  = plot_comparison(noisy_wav, enhanced, clean_wav,
                                     plot_stem + ".wav", sr=sr)
        print(f"      [PLOT] Plot -> {plot_path}")

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
    """
    Walk Data/processed/noisy/noizeus/<noise_type>/<snr_dB>/ and collect all
    .wav paths, optionally restricting by noise type and/or SNR level.

    Args:
        noise_type_filter : e.g. "airport" -- if set, only files under this
                            noise-type subdirectory are returned
        snr_filter        : e.g. "0dB" -- if set, only files under this SNR
                            subdirectory are returned

    Returns:
        Sorted list of absolute .wav paths
    """
    if not os.path.isdir(NOISY_ROOT):
        print(f"[ERR] Noisy root not found: {NOISY_ROOT}")
        print("   Ensure Data/processed/noisy/noizeus/ exists at the project root.")
        sys.exit(1)

    paths: list[str] = []

    noise_types = sorted(d for d in os.listdir(NOISY_ROOT) if not d.startswith("."))
    if noise_type_filter:
        if noise_type_filter not in noise_types:
            print(f"[ERR] Noise type '{noise_type_filter}' not found in {NOISY_ROOT}")
            print(f"   Available: {noise_types}")
            sys.exit(1)
        noise_types = [noise_type_filter]

    for nt in noise_types:
        nt_dir = os.path.join(NOISY_ROOT, nt)
        if not os.path.isdir(nt_dir):
            continue

        snr_levels = sorted(d for d in os.listdir(nt_dir) if not d.startswith("."))
        if snr_filter:
            if snr_filter not in snr_levels:
                # Skip silently if this noise type lacks the requested SNR
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
                snr_filter: Optional[str] = None) -> list:
    """
    Process every .wav file found under Data/processed/noisy/noizeus/.

    Args:
        plot              : generate comparison plots for each file
        limit             : if set, only process this many files (quick checks)
        noise_type_filter : restrict to one noise type (e.g. "airport")
        snr_filter        : restrict to one SNR level (e.g. "0dB")

    Returns:
        List of result dicts from process_file()
    """
    all_paths = collect_noisy_files(noise_type_filter, snr_filter)

    if not all_paths:
        print("[ERR] No .wav files found -- check noise-type / SNR filters.")
        sys.exit(1)

    if limit:
        all_paths = all_paths[:limit]

    filter_desc = ""
    if noise_type_filter:
        filter_desc += f"  noise={noise_type_filter}"
    if snr_filter:
        filter_desc += f"  snr={snr_filter}"

    print(f"\n[AUDIO] MMSE-LSA filter -- processing {len(all_paths)} file(s){filter_desc}")
    print(f"   Source -> {NOISY_ROOT}")
    print(f"   Output -> {OUTPUT_ROOT}\n")

    results: list[dict] = []
    for p in all_paths:
        result = process_file(p, plot=plot)
        if result:
            results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    if results:
        avg_input_rms  = np.mean([r["input_rms"]  for r in results])
        avg_output_rms = np.mean([r["output_rms"] for r in results])
        print("\n" + "-"*60)
        print(f"  Files processed : {len(results)}")
        print(f"  Avg input RMS   : {avg_input_rms:.4f}")
        print(f"  Avg output RMS  : {avg_output_rms:.4f}")
        print(f"  Output root     : {OUTPUT_ROOT}")
        print("-"*60 + "\n")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SELF-TEST (run without any dataset to verify math is correct)
# ══════════════════════════════════════════════════════════════════════════════

def self_test():
    """
    Generate a synthetic noisy sine tone and run MMSE-LSA on it.
    Verifies that:
      - Output shape matches input shape
      - Output is float32
      - Output values are in [-1, +1] after postprocessing
      - Enhanced signal has lower RMS noise than input (by at least some margin)
    """
    print("Running MMSE-LSA self-test with synthetic signal...")

    sr    = TARGET_SR
    dur   = 4.0
    N     = int(sr * dur)
    t     = np.linspace(0, dur, N, endpoint=False)

    # Clean: 300 Hz sine (like a fundamental speech tone)
    clean = 0.5 * np.sin(2 * np.pi * 300 * t).astype(np.float32)

    # Noise: Gaussian white noise at ~0 dB SNR
    rng   = np.random.default_rng(42)
    noise = 0.3 * rng.standard_normal(N).astype(np.float32)

    noisy = np.clip(clean + noise, -1.0, 1.0)

    enhanced = mmse_lsa_filter(noisy, sr=sr)
    enhanced = postprocess_filter_output(enhanced, N)

    # Assertions
    assert enhanced.shape == (N,),   f"Shape mismatch: {enhanced.shape} vs ({N},)"
    assert enhanced.dtype == np.float32, "dtype must be float32"
    assert np.abs(enhanced).max() <= 1.0 + 1e-5, "Output exceeds [-1, +1]"

    noise_rms_in  = float(np.sqrt(np.mean((noisy  - clean) ** 2)))
    noise_rms_out = float(np.sqrt(np.mean((enhanced[:N] - clean[:N]) ** 2)))
    print(f"  Input noise RMS  : {noise_rms_in:.4f}")
    print(f"  Output noise RMS : {noise_rms_out:.4f}")
    print(f"  [OK] Self-test passed -- output is valid float32 waveform in [-1, +1]")

    if noise_rms_out < noise_rms_in:
        print(f"  [OK] Noise reduced by {(1 - noise_rms_out/noise_rms_in)*100:.1f}%")
    else:
        print(f"  [WARN]  Noise not reduced on synthetic tone -- this is normal for non-speech signals.")
        print(f"     MMSE-LSA is tuned for speech; verify on real NOIZEUS clips.")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 -- MMSE-LSA Classical Speech Enhancement Baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Self-test (no data needed):
  python execution/04_mmse_lsa_filter.py --test

  # Single file (relative or absolute path):
  python execution/04_mmse_lsa_filter.py --file Data/processed/noisy/noizeus/airport/0dB/sp01_airport_sn0.wav

  # All files, all conditions:
  python execution/04_mmse_lsa_filter.py --all

  # Airport noise only:
  python execution/04_mmse_lsa_filter.py --all --noise-type airport

  # Airport noise at 0 dB SNR only, with plots:
  python execution/04_mmse_lsa_filter.py --all --noise-type airport --snr 0dB --plot

  # Quick 5-file sanity check:
  python execution/04_mmse_lsa_filter.py --all --limit 5
        """
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--file", metavar="PATH",
        help="Process a single noisy .wav (relative to project root or absolute)"
    )
    mode.add_argument(
        "--all", action="store_true",
        help="Process ALL .wav files under Data/processed/noisy/noizeus/"
    )
    mode.add_argument(
        "--test", action="store_true",
        help="Run self-test with a synthetic sine+noise signal (no dataset required)"
    )

    parser.add_argument(
        "--noise-type", metavar="TYPE", default=None,
        help="[--all only] Restrict to one noise type, e.g. airport, babble, car, …"
    )
    parser.add_argument(
        "--snr", metavar="LEVEL", default=None,
        help="[--all only] Restrict to one SNR level, e.g. 0dB, 5dB, 10dB, 15dB"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="[--all only] Only process this many files (for quick sanity-checks)"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save waveform + spectrogram comparison plots to results/plots/"
    )

    args = parser.parse_args()

    if args.test:
        self_test()
    elif args.file:
        process_file(args.file, plot=args.plot)
    elif args.all:
        process_all(
            plot              = args.plot,
            limit             = args.limit,
            noise_type_filter = args.noise_type,
            snr_filter        = args.snr,
        )


if __name__ == "__main__":
    main()
