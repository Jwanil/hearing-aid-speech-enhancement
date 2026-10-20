# execution/phase5_unet_film/26_eval_phase5a.py
# Authoritative Phase 5A Evaluation & Personalization Verification Suite
import os, sys, json, time, importlib.util
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pystoi import stoi

# Load model and dataset modules
HERE = os.path.dirname(__file__)

def load_mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

model_mod = load_mod("model", "20_model_unet_film.py")
data_mod = load_mod("data", "03c_data_pipeline_v2.py")

CausalUNetCRM = model_mod.CausalUNetCRM
HearingAidDatasetV2 = data_mod.HearingAidDatasetV2
build_filelists = data_mod.build_filelists
validate_filelists = data_mod.validate_filelists

def si_sdr(est, ref):
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    eps = 1e-8
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    s_target = alpha * ref
    e_noise = est - s_target
    return float(10 * np.log10((np.dot(s_target, s_target) + eps) / (np.dot(e_noise, e_noise) + eps)))

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval Phase 5A] Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    ckpt_path = "results/phase5_unet_film/checkpoints/best_composite.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CausalUNetCRM(n_fft=256).to(device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    best_epoch = ckpt.get("epoch", "unknown")
    best_val_loss = ckpt.get("val_loss", "unknown")
    print(f"[Eval Phase 5A] Loaded {ckpt_path} (trained epoch {best_epoch}, best val loss {best_val_loss:.4f})")

    # Load dataset
    clean, noise, rir = build_filelists("data/processed")
    counts = validate_filelists(clean, noise, rir)
    print(f"[Data] clean={counts['clean']} noise={counts['noise']} rir={counts['rir']}")

    audiograms_path = "results/phase1_audiology/audiograms.json"
    dataset = HearingAidDatasetV2(clean, noise, rir, audiograms_path, segment=64000, sr=16000)

    # 3 Held-out clinical test profiles
    profiles = {
        "Profile_A_Normal": torch.tensor([10, 10, 10, 10, 10, 10], dtype=torch.float32) / 120.0,
        "Profile_B_Presbycusis": torch.tensor([10, 15, 20, 45, 70, 85], dtype=torch.float32) / 120.0,
        "Profile_C_SevereFlat": torch.tensor([60, 60, 60, 60, 60, 60], dtype=torch.float32) / 120.0,
    }

    n_eval = 100  # 100 test samples
    results_by_profile = {p: {"si_sdr_in": [], "si_sdr_out": [], "stoi_in": [], "stoi_out": []} for p in profiles}
    m_mag_max = 0.0
    diff_AB_list = []
    diff_BC_list = []

    print(f"[Eval Phase 5A] Running evaluation over {n_eval} samples across 3 audiogram profiles...")
    t0 = time.time()

    with torch.no_grad():
        for i in range(n_eval):
            sample = dataset[i]
            clean_wav = sample["clean"].numpy().squeeze(0)  # (64000,)
            noisy_wav = sample["noisy"].numpy().squeeze(0)
            noisy_t = sample["noisy"].unsqueeze(0).to(device)  # (1, 1, 64000)

            sdr_in = si_sdr(noisy_wav, clean_wav)
            st_in = stoi(clean_wav, noisy_wav, 16000, extended=False)

            enh_wavs = {}
            for pname, ag in profiles.items():
                ag_t = ag.unsqueeze(0).to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    enh_t, (m_real, m_imag, m_mag, m_phase), _ = model(noisy_t, ag_t)
                
                enh = enh_t.squeeze().float().cpu().numpy()
                enh_wavs[pname] = enh
                m_mag_max = max(m_mag_max, float(m_mag.max().cpu()))

                sdr_out = si_sdr(enh, clean_wav)
                st_out = stoi(clean_wav, enh, 16000, extended=False)

                results_by_profile[pname]["si_sdr_in"].append(sdr_in)
                results_by_profile[pname]["si_sdr_out"].append(sdr_out)
                results_by_profile[pname]["stoi_in"].append(st_in)
                results_by_profile[pname]["stoi_out"].append(st_out)

            # Measure personalization difference between profiles on same audio
            diff_AB = np.mean(np.abs(enh_wavs["Profile_A_Normal"] - enh_wavs["Profile_B_Presbycusis"]))
            diff_BC = np.mean(np.abs(enh_wavs["Profile_B_Presbycusis"] - enh_wavs["Profile_C_SevereFlat"]))
            diff_AB_list.append(diff_AB)
            diff_BC_list.append(diff_BC)

            if (i + 1) % 25 == 0:
                print(f"  Processed {i+1}/{n_eval} samples ({time.time() - t0:.1f}s)")

    print("\n" + "=" * 70)
    print("                PHASE 5A COMPREHENSIVE BENCHMARK REPORT")
    print("=" * 70)
    print(f"Model: CausalUNetCRM (812k params) | Checkpoint: best_composite.pt (Epoch {best_epoch})")
    print(f"Max Mask Magnitude (|M|): {m_mag_max:.4f} (<= 1.0 Strict Invariant: {m_mag_max <= 1.0})")
    print("-" * 70)

    summary = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "m_mag_max": float(m_mag_max),
        "mask_invariant_passed": bool(m_mag_max <= 1.0),
        "mean_diff_profile_A_vs_B": float(np.mean(diff_AB_list)),
        "mean_diff_profile_B_vs_C": float(np.mean(diff_BC_list)),
        "profiles": {}
    }

    for pname, data in results_by_profile.items():
        sdr_in_mean = np.mean(data["si_sdr_in"])
        sdr_out_mean = np.mean(data["si_sdr_out"])
        sdr_gain = sdr_out_mean - sdr_in_mean

        stoi_in_mean = np.mean(data["stoi_in"])
        stoi_out_mean = np.mean(data["stoi_out"])
        stoi_gain = stoi_out_mean - stoi_in_mean

        summary["profiles"][pname] = {
            "si_sdr_input": float(sdr_in_mean),
            "si_sdr_enhanced": float(sdr_out_mean),
            "si_sdr_gain": float(sdr_gain),
            "stoi_input": float(stoi_in_mean),
            "stoi_enhanced": float(stoi_out_mean),
            "stoi_gain": float(stoi_gain),
        }

        print(f"\n{pname}:")
        print(f"  SI-SDR: {sdr_in_mean:.2f} dB -> {sdr_out_mean:.2f} dB  (Gain: +{sdr_gain:.2f} dB)")
        print(f"  STOI:   {stoi_in_mean:.4f} -> {stoi_out_mean:.4f}  (Gain: +{stoi_gain:.4f})")

    print("\n" + "-" * 70)
    print("PERSONALIZATION INVARIANT TEST:")
    print(f"  Mean Output Waveform Diff (Profile A vs B): {np.mean(diff_AB_list):.6f}")
    print(f"  Mean Output Waveform Diff (Profile B vs C): {np.mean(diff_BC_list):.6f}")
    print(f"  Personalization Active: {np.mean(diff_AB_list) > 1e-4}")
    print("=" * 70)

    # Save to JSON
    out_json = "results/phase5_unet_film/phase5a_eval_results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved full benchmark results to {out_json}")

if __name__ == "__main__":
    evaluate()
