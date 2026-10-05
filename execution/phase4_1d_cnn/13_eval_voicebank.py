"""
execution/phase4_1d_cnn/13_eval_voicebank.py
============================================
Phase 4 Evaluation on local VoiceBank-DEMAND test set.
Uses: data/processed/voicebank/clean_testset_wav/
      data/processed/voicebank/noisy_testset_wav/

Same domain as training -> correct evaluation protocol.
Published full-training Conv-TasNet: ~+14 dB SI-SDR improvement.
Our 19-epoch partial run will be lower but directionally valid.
"""
import os, importlib.util
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm.auto import tqdm
from pystoi import stoi

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_spec = importlib.util.spec_from_file_location(
    "model_1d_cnn",
    os.path.join(os.path.dirname(__file__), "10_model_1d_cnn.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
get_1d_cnn_model = _mod.get_1d_cnn_model

CHECKPOINT_PATH = os.path.join(_root, "results", "checkpoints", "1d_cnn_best.pt")
CLEAN_DIR       = os.path.join(_root, "data", "processed", "voicebank", "clean_testset_wav")
NOISY_DIR       = os.path.join(_root, "data", "processed", "voicebank", "noisy_testset_wav")
RESULTS_CSV     = os.path.join(_root, "results", "phase4_voicebank_eval.csv")
SR              = 16000
SEGMENT_LEN     = 32000  # 2s — matches training crop

def si_sdr(est, ref):
    ref = ref - ref.mean(); est = est - est.mean()
    eps = 1e-8
    s_t = (np.dot(est, ref) / (np.dot(ref, ref) + eps)) * ref
    e_n = est - s_t
    return float(10 * np.log10((np.dot(s_t, s_t) + eps) / (np.dot(e_n, e_n) + eps)))

def load_wav(path):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(0).numpy()
    if len(wav) > SEGMENT_LEN:   wav = wav[:SEGMENT_LEN]
    elif len(wav) < SEGMENT_LEN: wav = np.pad(wav, (0, SEGMENT_LEN - len(wav)))
    return wav

def safe_stoi(ref, est, sr=SR):
    try:    return stoi(ref, est, sr, extended=False)
    except: return float("nan")

def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint not found: {CHECKPOINT_PATH}"); return

    device = torch.device("cpu")
    model  = get_1d_cnn_model()
    ck     = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: epoch {ck.get('epoch','?')}, train loss {ck.get('loss',0):.3f} dB")

    # Match noisy files to clean files by filename stem
    noisy_files = sorted(f for f in os.listdir(NOISY_DIR)
                         if f.endswith(".wav") and not f.startswith("._"))
    print(f"Test files: {len(noisy_files)}")

    results = []
    for fname in tqdm(noisy_files, desc="Evaluating"):
        noisy_path = os.path.join(NOISY_DIR, fname)
        clean_path = os.path.join(CLEAN_DIR, fname)  # same filename
        if not os.path.exists(clean_path):
            continue

        c = load_wav(clean_path)
        n = load_wav(noisy_path)

        t_in = torch.tensor(n, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            t_out = model(t_in)
        a = t_out.squeeze().cpu().numpy()
        ml = min(len(c), len(a))
        c, n, a = c[:ml], n[:ml], a[:ml]

        results.append({
            "file":        fname,
            "speaker":     fname[:4],
            "stoi_noisy":  safe_stoi(c, n),
            "sisdr_noisy": si_sdr(n, c),
            "stoi_cnn":    safe_stoi(c, a),
            "sisdr_cnn":   si_sdr(a, c),
        })

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved {len(df)} rows -> {RESULTS_CSV}")

    print("\n" + "="*58)
    print(f"{'Metric':<8} | {'Noisy baseline':>14} | {'CNN (Ep19)':>10} | Delta")
    print("-"*58)
    for metric, col in [("STOI","stoi"), ("SI-SDR","sisdr")]:
        nm = df[f"{col}_noisy"].mean()
        am = df[f"{col}_cnn"].mean()
        delta = am - nm
        sign  = "+" if delta >= 0 else ""
        print(f"{metric:<8} | {nm:>14.3f} | {am:>10.3f} | {sign}{delta:.3f}")
    print("="*58)
    print("\nReference: Published full Conv-TasNet => SI-SDR improvement ~+14 dB")
    print(f"Our 19-epoch partial: see delta above")

if __name__ == "__main__":
    main()
