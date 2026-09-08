#!/usr/bin/env python3
"""
execution/06_baseline_eval.py
─────────────────────────────────────────────────────────────────────────────
Phase 3 — Classical Baseline Evaluation

Computes STOI, PESQ (wideband), and SI-SDR for:
  1. Noisy input  (lower bound — no processing)
  2. MMSE-LSA enhanced
  3. Wavelet DWT enhanced

Across all NOIZEUS files: 8 noise types × 4 SNRs × 30 speakers

Usage:
  uv run --with pesq --with pystoi --with soundfile --with matplotlib \
         --python 3.14 python execution/06_baseline_eval.py

Output:
  results/classical_baselines.csv
  results/plots/baseline_stoi.png
  results/plots/baseline_pesq.png
  results/plots/baseline_sisdr.png
  results/plots/baseline_stoi_vs_snr.png
  results/cli_output/06_baseline_eval.txt
"""

import os, sys, csv, time, logging
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from pystoi import stoi as compute_stoi
except ImportError:
    print("ERROR: pystoi missing. Run with: uv run --with pystoi ..."); sys.exit(1)

try:
    from pesq import pesq as compute_pesq
except ImportError:
    print("ERROR: pesq missing. Run with: uv run --with pesq ..."); sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent

DATA_ROOTS = [
    Path("/Volumes/SANDISK/Minor Project/Data/processed"),
    PROJECT_ROOT / "data" / "processed",
    PROJECT_ROOT / "dataset" / "processed",
]
DATA_ROOT = next((p for p in DATA_ROOTS if p.exists()), None)
if DATA_ROOT is None:
    print("ERROR: Cannot find processed data. Mount the SSD."); sys.exit(1)

CLEAN_DIR   = DATA_ROOT / "clean"  / "noizeus" / "clean"
NOISY_DIR   = DATA_ROOT / "noisy"  / "noizeus"
MMSE_DIR    = PROJECT_ROOT / "results" / "enhanced_mmse"     / "noizeus"
WAVELET_DIR = PROJECT_ROOT / "results" / "enhanced_wavelet"  / "noizeus"
OUT_CSV     = PROJECT_ROOT / "results" / "classical_baselines.csv"
PLOTS_DIR   = PROJECT_ROOT / "results" / "plots"
CLI_LOG     = PROJECT_ROOT / "results" / "cli_output" / "06_baseline_eval.txt"
SR          = 16000

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
CLI_LOG.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(CLI_LOG, mode="w")],
)
log = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_wav(path):
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if sr != SR:
            log.warning(f"SR mismatch {sr} Hz: {path.name}"); return None
        return audio[:, 0] if audio.ndim > 1 else audio
    except Exception as e:
        log.debug(f"Load failed {path}: {e}"); return None

def align(ref, est):
    if len(est) > len(ref): return ref, est[:len(ref)]
    if len(est) < len(ref): return ref, np.pad(est, (0, len(ref)-len(est)))
    return ref, est

def safe_stoi(ref, est):
    try:
        ref, est = align(ref, est)
        return float(compute_stoi(ref, est, SR, extended=False))
    except: return float("nan")

def safe_pesq(ref, est):
    try:
        ref, est = align(ref, est)
        if len(ref) < SR * 0.5: return float("nan")
        return float(compute_pesq(SR, ref, est, "wb"))
    except: return float("nan")

def safe_sisdr(ref, est):
    try:
        ref, est = align(ref, est)
        ref = ref - ref.mean(); est = est - est.mean()
        alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-8)
        proj  = alpha * ref
        noise = est - proj
        return float(10.0 * np.log10(np.dot(proj,proj) / (np.dot(noise,noise)+1e-8)+1e-8))
    except: return float("nan")

def nanmean(vals):
    v = [x for x in vals if x == x]  # filter NaN
    return sum(v)/len(v) if v else float("nan")

def find_clean(speaker_id):
    c = CLEAN_DIR / f"{speaker_id}.wav"
    if c.exists(): return c
    hits = [f for f in CLEAN_DIR.iterdir()
            if f.name.startswith(speaker_id) and f.suffix==".wav" and not f.name.startswith("._")]
    return hits[0] if hits else None

# ── Main loop ─────────────────────────────────────────────────────────────────

def evaluate_all():
    log.info("="*70)
    log.info("Phase 3 — Classical Baseline Evaluation")
    log.info(f"Data: {DATA_ROOT}")
    log.info("="*70)

    rows = []
    noise_types = sorted(d.name for d in NOISY_DIR.iterdir()
                         if d.is_dir() and not d.name.startswith("."))

    for noise_type in noise_types:
        snr_dirs = sorted(d for d in (NOISY_DIR/noise_type).iterdir()
                          if d.is_dir() and not d.name.startswith("."))
        for snr_dir in snr_dirs:
            snr_label = snr_dir.name
            snr_val   = int(snr_label.replace("dB",""))
            noisy_files = sorted(f for f in snr_dir.iterdir()
                                 if f.suffix==".wav" and not f.name.startswith("._"))
            for noisy_path in noisy_files:
                spk = noisy_path.name.split("_")[0]
                clean_path = find_clean(spk)
                if clean_path is None: continue
                clean = load_wav(clean_path);  noisy = load_wav(noisy_path)
                if clean is None or noisy is None: continue

                mmse_path    = MMSE_DIR    / noise_type / snr_label / noisy_path.name
                wavelet_path = WAVELET_DIR / noise_type / snr_label / noisy_path.name
                mmse    = load_wav(mmse_path)    if mmse_path.exists()    else None
                wavelet = load_wav(wavelet_path) if wavelet_path.exists() else None

                row = {
                    "speaker": spk, "noise_type": noise_type,
                    "snr_label": snr_label, "snr_db": snr_val,
                    "noisy_stoi":  safe_stoi(clean, noisy),
                    "noisy_pesq":  safe_pesq(clean, noisy),
                    "noisy_sisdr": safe_sisdr(clean, noisy),
                    "mmse_stoi":   safe_stoi(clean, mmse)    if mmse    is not None else float("nan"),
                    "mmse_pesq":   safe_pesq(clean, mmse)    if mmse    is not None else float("nan"),
                    "mmse_sisdr":  safe_sisdr(clean, mmse)   if mmse    is not None else float("nan"),
                    "wav_stoi":    safe_stoi(clean, wavelet)  if wavelet is not None else float("nan"),
                    "wav_pesq":    safe_pesq(clean, wavelet)  if wavelet is not None else float("nan"),
                    "wav_sisdr":   safe_sisdr(clean, wavelet) if wavelet is not None else float("nan"),
                }
                rows.append(row)
                log.info(
                    f"[{noise_type:12s} {snr_label:4s}] {spk} | "
                    f"STOI  noisy={row['noisy_stoi']:.3f} mmse={row['mmse_stoi']:.3f} wav={row['wav_stoi']:.3f} | "
                    f"PESQ  noisy={row['noisy_pesq']:.2f} mmse={row['mmse_pesq']:.2f} wav={row['wav_pesq']:.2f} | "
                    f"SI-SDR noisy={row['noisy_sisdr']:.1f} mmse={row['mmse_sisdr']:.1f} wav={row['wav_sisdr']:.1f}"
                )
    return rows

# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(rows):
    log.info(f"\n{'='*70}")
    log.info("OVERALL MEAN (all noise types, all SNRs)")
    log.info(f"{'='*70}")
    log.info(f"{'Metric':<10} {'Noisy':>8} {'MMSE-LSA':>10} {'Wavelet':>10}  {'MMSE Δ':>8}  {'Wav Δ':>8}")
    log.info("-"*70)
    for label, cn, cm, cw in [
        ("STOI",   "noisy_stoi",  "mmse_stoi",  "wav_stoi"),
        ("PESQ",   "noisy_pesq",  "mmse_pesq",  "wav_pesq"),
        ("SI-SDR", "noisy_sisdr", "mmse_sisdr", "wav_sisdr"),
    ]:
        n = nanmean([r[cn] for r in rows])
        m = nanmean([r[cm] for r in rows])
        w = nanmean([r[cw] for r in rows])
        log.info(f"  {label:<8} {n:>8.3f} {m:>10.3f} {w:>10.3f}  {m-n:>+8.3f}  {w-n:>+8.3f}")

    log.info(f"\n{'─'*70}")
    log.info("STOI BY NOISE TYPE")
    log.info(f"  {'Noise':<14} {'Noisy':>8} {'MMSE':>8} {'Wavelet':>8}  {'MMSE Δ':>8}  {'Wav Δ':>8}")
    for nt in sorted(set(r["noise_type"] for r in rows)):
        sub = [r for r in rows if r["noise_type"]==nt]
        n = nanmean([r["noisy_stoi"] for r in sub])
        m = nanmean([r["mmse_stoi"]  for r in sub])
        w = nanmean([r["wav_stoi"]   for r in sub])
        log.info(f"  {nt:<14} {n:>8.3f} {m:>8.3f} {w:>8.3f}  {m-n:>+8.3f}  {w-n:>+8.3f}")

    log.info(f"\n{'─'*70}")
    log.info("STOI BY SNR LEVEL")
    log.info(f"  {'SNR':<10} {'Noisy':>8} {'MMSE':>8} {'Wavelet':>8}")
    for snr in ["0dB","5dB","10dB","15dB"]:
        sub = [r for r in rows if r["snr_label"]==snr]
        if not sub: continue
        n = nanmean([r["noisy_stoi"] for r in sub])
        m = nanmean([r["mmse_stoi"]  for r in sub])
        w = nanmean([r["wav_stoi"]   for r in sub])
        log.info(f"  {snr:<10} {n:>8.3f} {m:>8.3f} {w:>8.3f}")

# ── Plots ─────────────────────────────────────────────────────────────────────

DARK_BG = "#0f0f1a"; PANEL_BG = "#1a1a2e"
C_NOISY="#e74c3c"; C_MMSE="#3498db"; C_WAV="#2ecc71"

def bar_chart(rows, cn, cm, cw, title, ylabel, fname):
    noise_types = sorted(set(r["noise_type"] for r in rows))
    n_v = [nanmean([r[cn] for r in rows if r["noise_type"]==nt]) for nt in noise_types]
    m_v = [nanmean([r[cm] for r in rows if r["noise_type"]==nt]) for nt in noise_types]
    w_v = [nanmean([r[cw] for r in rows if r["noise_type"]==nt]) for nt in noise_types]

    x = np.arange(len(noise_types)); wd = 0.25
    fig, ax = plt.subplots(figsize=(13,5))
    fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(PANEL_BG)

    bn = ax.bar(x-wd, n_v, wd, label="Noisy Input",  color=C_NOISY, alpha=0.9)
    bm = ax.bar(x,    m_v, wd, label="MMSE-LSA",     color=C_MMSE,  alpha=0.9)
    bw = ax.bar(x+wd, w_v, wd, label="Wavelet DWT",  color=C_WAV,   alpha=0.9)

    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Noise Type", color="#aaaacc")
    ax.set_ylabel(ylabel, color="#aaaacc")
    ax.set_xticks(x); ax.set_xticklabels(noise_types, rotation=20, ha="right", color="white", fontsize=9)
    ax.tick_params(colors="white"); ax.spines[:].set_color("#444466")
    ax.legend(facecolor=PANEL_BG, edgecolor="#444466", labelcolor="white", fontsize=10)
    ax.grid(axis="y", color="#333355", linestyle="--", alpha=0.5)
    for bar in [*bn, *bm, *bw]:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.005,
                f"{h:.2f}", ha="center", va="bottom", color="white", fontsize=7)
    plt.tight_layout()
    out = PLOTS_DIR / fname
    plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(); log.info(f"Plot → {out}")

def line_stoi_vs_snr(rows):
    snrs = ["0dB","5dB","10dB","15dB"]; xs = [0,5,10,15]
    n_v = [nanmean([r["noisy_stoi"] for r in rows if r["snr_label"]==s]) for s in snrs]
    m_v = [nanmean([r["mmse_stoi"]  for r in rows if r["snr_label"]==s]) for s in snrs]
    w_v = [nanmean([r["wav_stoi"]   for r in rows if r["snr_label"]==s]) for s in snrs]

    fig, ax = plt.subplots(figsize=(8,5))
    fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(PANEL_BG)
    ax.plot(xs, n_v, "o-", color=C_NOISY, lw=2, ms=8, label="Noisy Input")
    ax.plot(xs, m_v, "s-", color=C_MMSE,  lw=2, ms=8, label="MMSE-LSA")
    ax.plot(xs, w_v, "^-", color=C_WAV,   lw=2, ms=8, label="Wavelet DWT")
    ax.set_title("STOI vs Input SNR — Classical Baselines", color="white", fontsize=13, fontweight="bold")
    ax.set_xlabel("Input SNR (dB)", color="#aaaacc")
    ax.set_ylabel("STOI", color="#aaaacc")
    ax.set_xticks(xs); ax.tick_params(colors="white"); ax.spines[:].set_color("#444466")
    ax.legend(facecolor=PANEL_BG, edgecolor="#444466", labelcolor="white")
    ax.grid(color="#333355", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out = PLOTS_DIR / "baseline_stoi_vs_snr.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(); log.info(f"Plot → {out}")

# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    log.info(f"Python {sys.version} | Root: {PROJECT_ROOT}")

    rows = evaluate_all()

    if rows:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        log.info(f"CSV → {OUT_CSV}  ({len(rows)} rows)")

        print_summary(rows)

        bar_chart(rows, "noisy_stoi","mmse_stoi","wav_stoi",
                  "STOI by Noise Type — Classical Baselines","STOI (↑ better)","baseline_stoi.png")
        bar_chart(rows, "noisy_pesq","mmse_pesq","wav_pesq",
                  "PESQ-WB by Noise Type — Classical Baselines","PESQ WB (↑ better)","baseline_pesq.png")
        bar_chart(rows, "noisy_sisdr","mmse_sisdr","wav_sisdr",
                  "SI-SDR by Noise Type — Classical Baselines","SI-SDR dB (↑ better)","baseline_sisdr.png")
        line_stoi_vs_snr(rows)

    log.info(f"\nDone in {time.time()-t0:.1f}s")
