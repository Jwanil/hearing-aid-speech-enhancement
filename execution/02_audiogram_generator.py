from __future__ import annotations
"""
Phase 1 — Audiology: Audiogram Generator
==========================================
Directive: directives/01_audiology.md

What this script does:
  Generates and visualises synthetic audiograms covering the three test profiles
  the project will use for evaluation, plus random realistic audiograms drawn from
  clinical hearing-loss distributions.

  An audiogram plots HEARING THRESHOLD LEVEL (dB HL) vs FREQUENCY (Hz).
  - 0–25 dB HL  = Normal hearing
  - 26–40 dB HL = Mild loss
  - 41–55 dB HL = Moderate loss
  - 56–70 dB HL = Moderately-severe loss
  - 71–90 dB HL = Severe loss
  - >90 dB HL   = Profound loss

  The 6 standard audiometric test frequencies are:
      250, 500, 1000, 2000, 4000, 8000 Hz

  The three project evaluation profiles (from directives):
      1. Mild flat loss         — [15, 20, 25, 25, 30, 30]
      2. Moderate sloping loss  — [10, 15, 30, 50, 65, 80]   ← the classic presbycusis curve
      3. Severe high-freq loss  — [10, 10, 35, 65, 85, 90]

Usage:
    python execution/02_audiogram_generator.py
    python execution/02_audiogram_generator.py --n-random 20 --seed 42

Outputs:
    results/plots/01_audiograms_profiles.png   — the 3 project test profiles
    results/plots/01_audiograms_random.png     — random realistic audiograms
    results/data/audiograms.json               — all audiograms as JSON for downstream use
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(ROOT, "results", "plots")
DATA_DIR = os.path.join(ROOT, "results", "data")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ─── Standard audiometric frequencies ────────────────────────────────────────
FREQUENCIES = [250, 500, 1000, 2000, 4000, 8000]   # Hz

# ─── The 3 project test profiles ─────────────────────────────────────────────
# These will be used across all 5 model evaluations (Phases 3–6)
TEST_PROFILES = {
    "Mild Flat Loss":
        [15, 20, 25, 25, 30, 30],
    "Moderate Sloping Loss (Presbycusis)":
        [10, 15, 30, 50, 65, 80],
    "Severe High-Frequency Loss":
        [10, 10, 35, 65, 85, 90],
}

# ─── Severity bands (per WHO / ASHA classification) ──────────────────────────
SEVERITY_BANDS = [
    (0,  25,  "Normal",             "#2ECC71"),
    (25, 40,  "Mild",               "#F1C40F"),
    (40, 55,  "Moderate",           "#E67E22"),
    (55, 70,  "Mod-Severe",         "#E74C3C"),
    (70, 90,  "Severe",             "#8E44AD"),
    (90, 120, "Profound",           "#2C3E50"),
]

PROFILE_COLORS = ["#3498DB", "#E74C3C", "#2ECC71"]
PROFILE_MARKERS = ["o", "s", "^"]


# ─── Audiogram generation ─────────────────────────────────────────────────────
def generate_random_audiogram(rng: np.random.Generator) -> list[float]:
    """
    Generate a clinically realistic audiogram using a sloping model.

    Real-world hearing loss (especially age-related) tends to:
      - Start mild at low frequencies (250–1000 Hz)
      - Worsen significantly at high frequencies (4000–8000 Hz)
      - Have correlated adjacent frequencies (loss doesn't jump erratically)

    Strategy:
      1. Draw a low-frequency anchor (250 Hz threshold) from N(20, 10) clipped to [0, 40]
      2. Draw a high-frequency anchor (8000 Hz threshold) from N(60, 20) clipped to [low, 100]
      3. Interpolate linearly in log-frequency space
      4. Add small correlated jitter per frequency
    """
    low_thresh  = float(np.clip(rng.normal(20, 10),  0,  40))
    high_thresh = float(np.clip(rng.normal(65, 20), max(low_thresh, 25), 100))

    # Log-frequency interpolation (perceptually uniform)
    log_freqs  = np.log2(FREQUENCIES)
    log_f_min  = log_freqs[0]
    log_f_max  = log_freqs[-1]
    interp     = (log_freqs - log_f_min) / (log_f_max - log_f_min)
    base       = low_thresh + interp * (high_thresh - low_thresh)

    # Add correlated jitter (adjacent frequencies co-vary)
    jitter = np.cumsum(rng.normal(0, 3, len(FREQUENCIES)))
    jitter -= jitter.mean()

    levels = np.clip(base + jitter, 0, 110).tolist()
    return [round(v, 1) for v in levels]


def classify_severity(levels: list[float]) -> str:
    """Classify overall severity from mean of 1000 + 2000 + 4000 Hz (clinical convention)."""
    key_freqs = [FREQUENCIES.index(f) for f in [1000, 2000, 4000]]
    mean_hl = np.mean([levels[i] for i in key_freqs])
    for lo, hi, name, _ in SEVERITY_BANDS:
        if lo <= mean_hl < hi:
            return name
    return "Profound"


# ─── Plotting helpers ─────────────────────────────────────────────────────────
def _setup_audiogram_axes(ax: plt.Axes, title: str):
    """Configure axes to look like a standard clinical audiogram."""
    ax.set_xlim(200, 9500)
    ax.set_ylim(120, -10)       # Audiograms are plotted INVERTED (0 at top)
    ax.set_xscale("log")
    ax.set_xticks(FREQUENCIES)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Frequency (Hz)", fontsize=10)
    ax.set_ylabel("Hearing Threshold Level (dB HL)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    ax.axhline(25,  color="#F1C40F", linestyle=":", alpha=0.5, linewidth=1)
    ax.axhline(40,  color="#E67E22", linestyle=":", alpha=0.5, linewidth=1)
    ax.axhline(55,  color="#E74C3C", linestyle=":", alpha=0.5, linewidth=1)
    ax.axhline(70,  color="#8E44AD", linestyle=":", alpha=0.5, linewidth=1)
    ax.axhline(90,  color="#2C3E50", linestyle=":", alpha=0.5, linewidth=1)

    # Shade severity bands
    for lo, hi, name, color in SEVERITY_BANDS:
        ax.axhspan(lo, hi, alpha=0.04, color=color)
        ax.text(220, (lo + hi) / 2, name, fontsize=6.5, color=color,
                va="center", alpha=0.8)


def plot_test_profiles(profiles: dict[str, list[float]]):
    fig, ax = plt.subplots(figsize=(9, 6))
    _setup_audiogram_axes(ax, "Project Test Audiogram Profiles\n(used for all 5-model evaluation)")

    for (name, levels), color, marker in zip(profiles.items(), PROFILE_COLORS, PROFILE_MARKERS):
        severity = classify_severity(levels)
        ax.plot(FREQUENCIES, levels, color=color, marker=marker,
                linewidth=2, markersize=8, label=f"{name}  [{severity}]")
        # Annotate each point with its dB value
        for freq, lv in zip(FREQUENCIES, levels):
            ax.annotate(f"{lv:.0f}", (freq, lv),
                        textcoords="offset points", xytext=(0, -12),
                        ha="center", fontsize=7.5, color=color)

    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "01_audiograms_profiles.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✅  Saved: {path}")


def plot_random_audiograms(random_audiograms: list[dict]):
    n = len(random_audiograms)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.2))
    axes = axes.flatten() if n > 1 else [axes]

    for i, ag in enumerate(random_audiograms):
        ax = axes[i]
        _setup_audiogram_axes(ax, f"#{i+1} — {ag['severity']}")
        ax.plot(FREQUENCIES, ag["levels"], color="#4A90D9",
                marker="o", linewidth=2, markersize=6)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{n} Randomly Generated Realistic Audiograms", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "01_audiograms_random.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  ✅  Saved: {path}")


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 1 — Audiogram Generator")
    parser.add_argument("--n-random", type=int, default=8,
                        help="Number of random audiograms to generate (default: 8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  PHASE 1 — AUDIOGRAM GENERATOR")
    print("="*55)

    # 1. Plot the 3 project test profiles
    print("\n[1/3] Generating project evaluation profiles …")
    plot_test_profiles(TEST_PROFILES)

    # 2. Generate random audiograms
    print(f"\n[2/3] Generating {args.n_random} random realistic audiograms (seed={args.seed}) …")
    rng = np.random.default_rng(args.seed)
    random_audiograms = []
    for _ in range(args.n_random):
        levels   = generate_random_audiogram(rng)
        severity = classify_severity(levels)
        random_audiograms.append({
            "frequencies": FREQUENCIES,
            "levels":      levels,
            "severity":    severity,
        })
        print(f"    {severity:16s}  {[f'{v:.0f}' for v in levels]}")

    plot_random_audiograms(random_audiograms)

    # 3. Save everything to JSON
    print("\n[3/3] Saving audiograms to JSON …")
    output = {
        "frequencies_hz": FREQUENCIES,
        "note": "Audiogram levels in dB HL. Higher = worse hearing.",
        "test_profiles": [
            {"name": name, "levels": levels, "severity": classify_severity(levels)}
            for name, levels in TEST_PROFILES.items()
        ],
        "random_audiograms": random_audiograms,
    }
    json_path = os.path.join(DATA_DIR, "audiograms.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  ✅  Saved: {json_path}")

    # 4. Summary
    print("\n[Summary]")
    print(f"  Standard frequencies : {FREQUENCIES} Hz")
    print(f"  Test profiles        : {len(TEST_PROFILES)}")
    print(f"  Random audiograms    : {args.n_random}")
    print("\n  Severity bands (dB HL):")
    for lo, hi, name, _ in SEVERITY_BANDS:
        print(f"    {lo:>3}–{hi:>3}  →  {name}")
    print("\n  ℹ️  Key insight: Most hearing aids compensate by applying GAIN at")
    print("     frequencies where the user's threshold is elevated. Your model")
    print("     learns to do this automatically from the audiogram vector.")
    print("\n" + "="*55 + "\n")


if __name__ == "__main__":
    main()
