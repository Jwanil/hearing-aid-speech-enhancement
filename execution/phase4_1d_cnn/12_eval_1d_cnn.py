import os
import sys
import importlib.util
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pystoi import stoi
from pesq import pesq
from asteroid.losses import pairwise_neg_sisdr

# ── Import model via importlib (digit in filename workaround) ─────────────────
_spec = importlib.util.spec_from_file_location(
    "model_1d_cnn",
    os.path.join(os.path.dirname(__file__), "10_model_1d_cnn.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
get_1d_cnn_model = _mod.get_1d_cnn_model

# ── Import MMSE-LSA filter via importlib (digit in filename workaround) ───────
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_mmse_path = os.path.join(_root, "execution", "phase3_classical_baselines", "04_mmse_lsa_filter.py")
_spec2 = importlib.util.spec_from_file_location("mmse_lsa_filter", _mmse_path)
_mod2 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_mod2)
mmse_lsa_filter = _mod2.mmse_lsa_filter
postprocess_filter_output = _mod2.postprocess_filter_output

# ── Configuration ─────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "results/checkpoints/1d_cnn_best.pt"  # Symlink or copy best epoch here
CLEAN_DIR = "data/processed/clean/noizeus/clean"
NOISY_DIR = "data/processed/noisy/noizeus"
RESULTS_CSV = "results/1d_cnn_comparison.csv"
SR = 16000

# Synthetic audiogram — Profile B: severe high-frequency loss (ski-slope)
# Frequencies: 250, 500, 1k, 2k, 4k, 6k Hz
# Thresholds: 10, 15, 20, 45, 70, 85 dB HL
AUDIOGRAM_FREQ = np.array([250, 500, 1000, 2000, 4000, 6000])
AUDIOGRAM_LOSS = np.array([10,  15,  20,   45,   70,   85])

# ── SI-SDR helper ─────────────────────────────────────────────────────────────
def compute_sisdr(est: np.ndarray, ref: np.ndarray) -> float:
    """Scalar SI-SDR in dB (higher = better)."""
    est_t = torch.tensor(est, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,T)
    ref_t = torch.tensor(ref, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return -pairwise_neg_sisdr(est_t, ref_t).item()

# ── HASPI helper ──────────────────────────────────────────────────────────────
def compute_haspi(clean: np.ndarray, enhanced: np.ndarray) -> float:
    """
    HASPI v2 score (0-1, higher = better intelligibility for hearing-impaired).
    Requires pyclarity >= 0.3.
    Returns 0.0 and warns if pyclarity is unavailable.
    """
    try:
        from clarity.evaluator.haspi import haspi_v2
        score, _ = haspi_v2(clean, SR, enhanced, SR, AUDIOGRAM_FREQ, AUDIOGRAM_LOSS)
        return float(score)
    except Exception as e:
        return -1.0  # Sentinel: pyclarity not available or error

# ── Per-file evaluation ───────────────────────────────────────────────────────
def evaluate_file(model, clean_path: str, noisy_path: str, device):
    """
    Returns a dict of metrics for both Experiment A (raw CNN) and
    Experiment B (MMSE-LSA pre-filter + CNN) and the noisy baseline.
    """
    # Load both files — already standardized to 16kHz, mono, 64000 samples
    clean_wav, _ = torchaudio.load(clean_path)   # (1, 64000)
    noisy_wav, _ = torchaudio.load(noisy_path)   # (1, 64000)
    clean_np  = clean_wav.squeeze().numpy()
    noisy_np  = noisy_wav.squeeze().numpy()

    # ── Experiment A: CNN on raw noisy audio ──────────────────────────────────
    cnn_input_a = noisy_wav.unsqueeze(0).to(device)          # (1, 1, 64000)
    with torch.no_grad():
        enhanced_a_t = model(cnn_input_a)                    # (1, 1, 64000)
    enhanced_a = enhanced_a_t.squeeze().cpu().numpy()

    # ── Experiment B: MMSE-LSA → CNN cascade ─────────────────────────────────
    mmse_out    = mmse_lsa_filter(noisy_np)
    mmse_clean  = postprocess_filter_output(mmse_out, original_length=len(noisy_np))
    cnn_input_b = torch.tensor(mmse_clean, dtype=torch.float32
                                ).unsqueeze(0).unsqueeze(0).to(device)    # (1,1,64000)
    with torch.no_grad():
        enhanced_b_t = model(cnn_input_b)
    enhanced_b = enhanced_b_t.squeeze().cpu().numpy()

    # ── Compute all metrics ───────────────────────────────────────────────────
    def metrics(est: np.ndarray, label: str) -> dict:
        return {
            f'pesq_{label}':  pesq(SR, clean_np, est, 'wb'),
            f'stoi_{label}':  stoi(clean_np, est, SR, extended=False),
            f'sisdr_{label}': compute_sisdr(est, clean_np),
            f'haspi_{label}': compute_haspi(clean_np, est),
        }

    result = {}
    result.update(metrics(noisy_np,   'noisy'))
    result.update(metrics(enhanced_a, 'a'))
    result.update(metrics(enhanced_b, 'b'))
    return result

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint not found at {CHECKPOINT_PATH}")
        print("After Colab training finishes, copy the best epoch checkpoint there.")
        print("Example: cp results/checkpoints/1d_cnn_epoch_030.pt results/checkpoints/1d_cnn_best.pt")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = get_1d_cnn_model()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # Walk all NOIZEUS noisy files
    results = []
    noise_types = sorted(
        d for d in os.listdir(NOISY_DIR)
        if os.path.isdir(os.path.join(NOISY_DIR, d))
    )

    all_files = [
        (nt, snr, f)
        for nt in noise_types
        for snr in sorted(os.listdir(os.path.join(NOISY_DIR, nt)))
        if os.path.isdir(os.path.join(NOISY_DIR, nt, snr))
        for f in sorted(os.listdir(os.path.join(NOISY_DIR, nt, snr)))
        if f.endswith('.wav')
    ]
    print(f"Evaluating {len(all_files)} files (Exp A and B per file)...")

    for nt, snr, fname in tqdm(all_files):
        noisy_path = os.path.join(NOISY_DIR, nt, snr, fname)
        # NOIZEUS filename format: spXX_<noise>_snYY.wav → clean is spXX.wav
        speaker = fname.split('_')[0]  # e.g. "sp01"
        clean_path = os.path.join(CLEAN_DIR, f"{speaker}.wav")

        if not os.path.exists(clean_path):
            continue

        metrics = evaluate_file(model, clean_path, noisy_path, device)
        row = {'noise_type': nt, 'snr': snr.replace('sn', ''), 'speaker': speaker, 'filename': fname}
        row.update(metrics)
        results.append(row)

    # Save CSV
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n✅ Saved {len(df)} rows to {RESULTS_CSV}")

    # Summary table
    print("\n" + "="*65)
    print(f"{'Metric':<8} | {'Noisy':>8} | {'Exp A (CNN)':>12} | {'Exp B (MMSE+CNN)':>16}")
    print("-"*65)
    for m in ['haspi', 'stoi', 'pesq', 'sisdr']:
        noisy_m = df[f'{m}_noisy'].mean()
        a_m     = df[f'{m}_a'].mean()
        b_m     = df[f'{m}_b'].mean()
        winner  = 'A' if a_m > b_m else 'B'
        print(f"{m.upper():<8} | {noisy_m:>8.3f} | {a_m:>12.3f} | {b_m:>16.3f}  ← {winner} wins")

if __name__ == "__main__":
    main()
