"""Deterministic Phase 5A evaluation on the official VoiceBank test split.

The Phase A training loader intentionally performs dynamic mixing. This
evaluator must not reuse that loader because it would evaluate on training
clean utterances with a new random augmentation. Instead, it uses the fixed
official ``clean_testset_wav``/``noisy_testset_wav`` pairs, records a manifest
and provenance, and writes both aggregate JSON and per-file CSV results.
"""

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchaudio
from pystoi import stoi


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SR = 16_000
HOP = 64


def load_mod(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model_mod = load_mod("phase5_model", "20_model_unet_film.py")
data_mod = load_mod("phase5_data", "03c_data_pipeline_v2.py")
CausalUNetCRM = model_mod.CausalUNetCRM
build_voicebank_test_pairs = data_mod.build_voicebank_test_pairs


PROFILES = {
    "Profile_A_Normal": [10, 10, 10, 10, 10, 10],
    "Profile_B_Presbycusis": [10, 15, 20, 45, 70, 85],
    "Profile_C_SevereFlat": [60, 60, 60, 60, 60, 60],
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_state():
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return {"commit": revision, "tracked_tree_clean": not bool(status), "tracked_status": status}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "tracked_tree_clean": False, "tracked_status": "unavailable"}


def set_deterministic_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def si_sdr(est, ref):
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    eps = 1e-8
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    target = alpha * ref
    error = est - target
    return float(10 * np.log10((np.dot(target, target) + eps) / (np.dot(error, error) + eps)))


def load_mono_16k(path):
    wav, sample_rate = torchaudio.load(path)
    if sample_rate != SR:
        raise ValueError(f"Expected {SR} Hz audio, got {sample_rate} Hz: {path}")
    wav = wav.mean(dim=0) if wav.shape[0] > 1 else wav.squeeze(0)
    if wav.numel() == 0 or not torch.isfinite(wav).all():
        raise ValueError(f"Audio is empty or non-finite: {path}")
    return wav.float()


def relative_data_path(path, data_dir):
    return os.path.relpath(path, data_dir)


def write_manifest(pairs, data_dir, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("filename", "clean_path", "noisy_path", "clean_sha256", "noisy_sha256"))
    for clean_path, noisy_path in pairs:
        writer.writerow(
            (
                Path(clean_path).name,
                relative_data_path(clean_path, data_dir),
                relative_data_path(noisy_path, data_dir),
                sha256_file(clean_path),
                sha256_file(noisy_path),
            )
        )
    text = buffer.getvalue()
    path.write_text(text, encoding="utf-8")
    return sha256_text(text)


def profile_tensor(levels, device):
    return torch.tensor(levels, dtype=torch.float32, device=device).unsqueeze(0) / 120.0


def evaluate(args):
    set_deterministic_seed(args.seed)
    data_dir = (ROOT / args.data_dir).resolve()
    checkpoint_path = (ROOT / args.checkpoint).resolve()
    results_path = (ROOT / args.results_json).resolve()
    csv_path = (ROOT / args.results_csv).resolve()
    manifest_path = (ROOT / args.manifest).resolve()
    git = git_state()
    if not git["tracked_tree_clean"]:
        raise RuntimeError(
            "Tracked Git changes are present; commit the Phase 5A source before evaluation "
            "so the result provenance identifies the exact code."
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    pairs = build_voicebank_test_pairs(str(data_dir))
    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]
    if not pairs:
        raise ValueError("No evaluation pairs remain after applying --max_samples")
    manifest_hash = write_manifest(pairs, str(data_dir), manifest_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_hash = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CausalUNetCRM(n_fft=256).to(device).eval()
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)

    profiles = {name: profile_tensor(levels, device) for name, levels in PROFILES.items()}
    rows = []
    mask_max = 0.0
    diff_ab = []
    diff_bc = []
    started = time.perf_counter()

    for index, (clean_path, noisy_path) in enumerate(pairs):
        clean = load_mono_16k(clean_path)
        noisy = load_mono_16k(noisy_path)
        if clean.shape != noisy.shape:
            raise ValueError(
                f"Clean/noisy length mismatch for {Path(clean_path).name}: "
                f"{clean.shape[-1]} != {noisy.shape[-1]}"
            )
        noisy_batch = noisy.unsqueeze(0).unsqueeze(0).to(device)
        clean_np = clean.numpy()
        noisy_np = noisy.numpy()
        base = {
            "index": index,
            "file": Path(clean_path).name,
            "clean_path": relative_data_path(clean_path, str(data_dir)),
            "noisy_path": relative_data_path(noisy_path, str(data_dir)),
            "num_samples": int(clean.shape[-1]),
            "si_sdr_input": si_sdr(noisy_np, clean_np),
            "stoi_input": float(stoi(clean_np, noisy_np, SR, extended=False)),
        }
        outputs = {}
        with torch.inference_mode():
            for profile_name, audiogram in profiles.items():
                amp_enabled = device.type == "cuda"
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                    enhanced, (_, _, m_mag, _), _ = model(noisy_batch, audiogram)
                enhanced_np = enhanced.squeeze().float().cpu().numpy()
                outputs[profile_name] = enhanced_np
                mask_max = max(mask_max, float(m_mag.max().cpu()))
                row = dict(base)
                row.update(
                    {
                        "profile": profile_name,
                        "si_sdr_enhanced": si_sdr(enhanced_np, clean_np),
                        "stoi_enhanced": float(stoi(clean_np, enhanced_np, SR, extended=False)),
                    }
                )
                row["si_sdr_gain"] = row["si_sdr_enhanced"] - row["si_sdr_input"]
                row["stoi_gain"] = row["stoi_enhanced"] - row["stoi_input"]
                rows.append(row)
        diff_ab.append(float(np.mean(np.abs(outputs["Profile_A_Normal"] - outputs["Profile_B_Presbycusis"]))))
        diff_bc.append(float(np.mean(np.abs(outputs["Profile_B_Presbycusis"] - outputs["Profile_C_SevereFlat"]))))

    fieldnames = list(rows[0].keys())
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        # Use LF explicitly so the recorded artifact hash is stable across
        # Windows and POSIX checkouts.
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for profile_name in PROFILES:
        profile_rows = [row for row in rows if row["profile"] == profile_name]
        summary[profile_name] = {
            "si_sdr_input": float(np.mean([row["si_sdr_input"] for row in profile_rows])),
            "si_sdr_enhanced": float(np.mean([row["si_sdr_enhanced"] for row in profile_rows])),
            "si_sdr_gain": float(np.mean([row["si_sdr_gain"] for row in profile_rows])),
            "stoi_input": float(np.mean([row["stoi_input"] for row in profile_rows])),
            "stoi_enhanced": float(np.mean([row["stoi_enhanced"] for row in profile_rows])),
            "stoi_gain": float(np.mean([row["stoi_gain"] for row in profile_rows])),
        }

    results = {
        "schema_version": 2,
        "protocol": "official_voicebank_test_pairs",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git["commit"],
        "git_tracked_tree_clean": git["tracked_tree_clean"],
        "command": {
            "argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            "working_directory": str(ROOT),
        },
        "checkpoint": {
            "path": os.path.relpath(checkpoint_path, ROOT),
            "sha256": checkpoint_hash,
            "epoch": checkpoint.get("epoch", "unknown") if isinstance(checkpoint, dict) else "unknown",
            "val_loss": checkpoint.get("val_loss", "unknown") if isinstance(checkpoint, dict) else "unknown",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "dataset": {
            "data_dir": args.data_dir,
            "pair_count": len(pairs),
            "manifest": os.path.relpath(manifest_path, ROOT),
            "manifest_sha256": manifest_hash,
            "evaluation_split": "VoiceBank-DEMAND official clean_testset_wav/noisy_testset_wav",
        },
        "evaluation": {
            "seed": args.seed,
            "sample_rate": SR,
            "n_fft": 256,
            "hop_length": HOP,
            "center": True,
            "model_is_causal": False,
            "profile_vectors_db_hl": PROFILES,
            "elapsed_seconds": time.perf_counter() - started,
            "mask_max": mask_max,
            "mask_invariant_passed": bool(mask_max <= 1.0 + 1e-6),
            "mean_output_abs_diff_profile_A_vs_B": float(np.mean(diff_ab)),
            "mean_output_abs_diff_profile_B_vs_C": float(np.mean(diff_bc)),
            "personalization_signal_detected": bool(np.mean(diff_ab) > 1e-4),
        },
        "profiles": summary,
        "artifacts": {
            "per_sample_csv": os.path.relpath(csv_path, ROOT),
            "per_sample_csv_sha256": sha256_file(csv_path),
        },
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved aggregate results: {results_path}")
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved per-sample CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--checkpoint", default="results/phase5_unet_film/checkpoints/best_composite.pt")
    parser.add_argument("--results_json", default="results/phase5_unet_film/phase5a_eval_results.json")
    parser.add_argument("--results_csv", default="results/phase5_unet_film/phase5a_eval_per_sample.csv")
    parser.add_argument("--manifest", default="results/phase5_unet_film/phase5a_eval_manifest.csv")
    parser.add_argument("--max_samples", type=int, default=0, help="0 evaluates every official test pair")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
