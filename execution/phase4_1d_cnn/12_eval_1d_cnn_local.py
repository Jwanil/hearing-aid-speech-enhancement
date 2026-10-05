"""
execution/phase4_1d_cnn/12_eval_1d_cnn_local.py
================================================
Phase 4 Evaluation - Local Mac version (no pesq / asteroid required)

Experiment A: noisy -> Conv-TasNet CNN -> enhanced
Experiment B: MMSE-LSA output (pre-computed) -> Conv-TasNet CNN -> enhanced

Metrics: STOI (pystoi), SI-SDR (manual numpy)
"""

import os, sys, importlib.util
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm.auto import tqdm
from pystoi import stoi

# -- Import model via importlib (digit-in-filename workaround) -----------------
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_spec = importlib.util.spec_from_file_location(
    "model_1d_cnn",
    os.path.join(os.path.dirname(__file__), "10_model_1d_cnn.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
get_1d_cnn_model = _mod.get_1d_cnn_model

# -- Paths ---------------------------------------------------------------------
CHECKPOINT_PATH = os.path.join(_root, "results", "checkpoints", "1d_cnn_best.pt")
NOISY_ROOT      = os.path.join(_root, "data", "processed", "noisy", "noizeus")
CLEAN_DIR       = os.path.join(_root, "data", "processed", "clean", "noizeus", "clean")
MMSE_ROOT       = os.path.join(_root, "results", "enhanced_mmse", "noizeus")
RESULTS_CSV     = os.path.join(_root, "results", "phase4_eval_results.csv")
TARGET_LEN      = 32000  # pad/crop at native SR (8kHz = 4s)
CNN_SR          = 16000  # model was trained at 16kHz
NOIZEUS_SR      = 8000   # NOIZEUS native sample rate
# CNN sees 32000 samples @ 16kHz = 2 seconds (matches training segment length)

# -- SI-SDR (manual) -----------------------------------------------------------
def si_sdr(est, ref):
    ref = ref - ref.mean(); est = est - est.mean()
    eps = 1e-8
    s_t = (np.dot(est, ref) / (np.dot(ref, ref) + eps)) * ref
    e_n = est - s_t
    return float(10 * np.log10((np.dot(s_t, s_t) + eps) / (np.dot(e_n, e_n) + eps)))

# -- Audio helpers -------------------------------------------------------------
def load_wav(path, target_len=None):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(0).numpy()
    if target_len:
        if len(wav) > target_len:   wav = wav[:target_len]
        elif len(wav) < target_len: wav = np.pad(wav, (0, target_len - len(wav)))
    return wav, sr

def cnn_enhance(model, wav_np, device, native_sr=NOIZEUS_SR):
    """Resample to 16kHz (CNN training SR), run model, resample back to native SR."""
    t_native = torch.tensor(wav_np, dtype=torch.float32).unsqueeze(0)       # (1, T@8k)
    t_16k    = torchaudio.functional.resample(t_native, native_sr, CNN_SR)   # (1, T@16k)
    t_in     = t_16k.unsqueeze(0).to(device)                                 # (1,1,T@16k)
    with torch.no_grad():
        t_out = model(t_in)                                                   # (1,1,T@16k)
    t_8k = torchaudio.functional.resample(t_out.squeeze(0), CNN_SR, native_sr)  # (1, T@8k)
    return t_8k.squeeze().cpu().numpy()

# -- Main ----------------------------------------------------------------------
def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint not found: {CHECKPOINT_PATH}"); return

    device = torch.device("cpu")
    print(f"Device: {device}")

    model = get_1d_cnn_model()
    ck    = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint epoch {ck.get('epoch','?')}, loss {ck.get('loss',0):.3f} dB")

    all_files = [
        (nt, snr, f)
        for nt  in sorted(os.listdir(NOISY_ROOT))
        if os.path.isdir(os.path.join(NOISY_ROOT, nt))
        for snr in sorted(os.listdir(os.path.join(NOISY_ROOT, nt)))
        if os.path.isdir(os.path.join(NOISY_ROOT, nt, snr))
        for f   in sorted(os.listdir(os.path.join(NOISY_ROOT, nt, snr)))
        if f.endswith(".wav")
    ]
    print(f"Evaluating {len(all_files)} files (Exp A + B each)...")

    results = []
    for noise_type, snr_level, fname in tqdm(all_files):
        noisy_path = os.path.join(NOISY_ROOT, noise_type, snr_level, fname)
        mmse_path  = os.path.join(MMSE_ROOT,  noise_type, snr_level, fname)
        speaker    = fname.split("_")[0]
        clean_path = os.path.join(CLEAN_DIR, f"{speaker}.wav")
        if not os.path.exists(clean_path): continue

        clean_np, sr = load_wav(clean_path, TARGET_LEN)
        noisy_np, _  = load_wav(noisy_path, TARGET_LEN)
        L = min(len(clean_np), TARGET_LEN)
        c = clean_np[:L]; n = noisy_np[:L]

        # Exp A: noisy -> CNN
        a = cnn_enhance(model, n, device)[:L]

        # Exp B: MMSE -> CNN
        b = None
        if os.path.exists(mmse_path):
            mmse_np, _ = load_wav(mmse_path, TARGET_LEN)
            b = cnn_enhance(model, mmse_np[:L], device)[:L]

        def safe_stoi(ref, est):
            try:    return stoi(ref, est, sr, extended=False)
            except: return float("nan")

        results.append({
            "noise_type":  noise_type,
            "snr":         snr_level.replace("dB",""),
            "speaker":     speaker,
            "stoi_noisy":  safe_stoi(c, n),
            "sisdr_noisy": si_sdr(n, c),
            "stoi_a":      safe_stoi(c, a),
            "sisdr_a":     si_sdr(a, c),
            "stoi_b":      safe_stoi(c, b) if b is not None else float("nan"),
            "sisdr_b":     si_sdr(b, c)    if b is not None else float("nan"),
        })

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved {len(df)} rows -> {RESULTS_CSV}")

    # Summary
    print("\n" + "="*62)
    print(f"{'Metric':<8} | {'Noisy':>8} | {'Exp A (CNN)':>12} | {'Exp B (MMSE+CNN)':>16} | Winner")
    print("-"*62)
    for metric, col, fmt in [("STOI","stoi",".3f"),("SI-SDR","sisdr",".2f")]:
        nm = df[f"{col}_noisy"].mean()
        am = df[f"{col}_a"].mean()
        bm = df[f"{col}_b"].mean()
        w  = "B (cascade)" if bm > am else "A (raw CNN)"
        print(f"{metric:<8} | {nm:>8.3f} | {am:>12{fmt}} | {bm:>16{fmt}} | {w}")
    print("="*62)
    print("\nBy noise type (SI-SDR):")
    print(df.groupby("noise_type")[["sisdr_noisy","sisdr_a","sisdr_b"]].mean().round(2).to_string())

if __name__ == "__main__":
    main()
