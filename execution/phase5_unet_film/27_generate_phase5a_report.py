"""Generate the Phase 5A report from the provenance-rich result JSON."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def fmt(value, digits=4):
    return f"{value:.{digits}f}"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(root, relative_path, expected_hash, label):
    path = root / relative_path
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"{label} hash mismatch: JSON={expected_hash} actual={actual_hash} path={path}"
        )
    return path


def generate(results_path, report_path):
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if results.get("schema_version") != 2:
        raise ValueError(
            "The result JSON is not from the corrected Phase 5A evaluator. "
            "Run 26_eval_phase5a.py first."
        )

    checkpoint = results["checkpoint"]
    dataset = results["dataset"]
    evaluation = results["evaluation"]
    profiles = results["profiles"]
    checkpoint_path = verify_artifact(ROOT, checkpoint["path"], checkpoint["sha256"], "Checkpoint")
    manifest_path = verify_artifact(ROOT, dataset["manifest"], dataset["manifest_sha256"], "Manifest")
    csv_path = verify_artifact(
        ROOT,
        results["artifacts"]["per_sample_csv"],
        results["artifacts"]["per_sample_csv_sha256"],
        "Per-sample CSV",
    )
    lines = [
        "# Phase 5A Verification Report",
        "",
        "**Scope:** Phase 5A composite-loss foundation only",
        "**Model:** `CausalUNetCRM` class name retained for checkpoint compatibility",
        "**Phase B:** Not implemented in this branch",
        "",
        "## 1. Reproducibility record",
        "",
        f"- Git commit: `{results['git_commit']}`",
        f"- Git tracked tree clean: `{results['git_tracked_tree_clean']}`",
        f"- Checkpoint: `{checkpoint['path']}`",
        f"- Checkpoint SHA-256: `{checkpoint['sha256']}`",
        f"- Checkpoint epoch: `{checkpoint['epoch']}`",
        f"- Checkpoint validation loss: `{checkpoint['val_loss']}`",
        f"- Evaluation protocol: `{results['protocol']}`",
        f"- Evaluation pairs: `{dataset['pair_count']}`",
        f"- Manifest: `{dataset['manifest']}`",
        f"- Manifest SHA-256: `{dataset['manifest_sha256']}`",
        f"- Per-sample CSV: `{results['artifacts']['per_sample_csv']}`",
        f"- Per-sample CSV SHA-256: `{results['artifacts']['per_sample_csv_sha256']}`",
        f"- Evaluator command: `{json.dumps(results['command']['argv'])}`",
        f"- Seed: `{evaluation['seed']}`",
        f"- PyTorch: `{results['environment']['torch']}`; torchaudio: `{results['environment']['torchaudio']}`",
        f"- CUDA: `{results['environment']['cuda_version']}`; GPU: `{results['environment']['gpu']}`",
        "",
        "## 2. Held-out benchmark",
        "",
        "The evaluator uses the official non-overlapping VoiceBank-DEMAND "
        "`clean_testset_wav`/`noisy_testset_wav` pairs. It does not call the "
        "dynamic training loader and does not add random noise, RIR, crop, or SNR augmentation.",
        "",
        "| Profile | SI-SDR input | SI-SDR enhanced | SI-SDR gain | STOI input | STOI enhanced | STOI gain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in profiles.items():
        lines.append(
            "| "
            + " | ".join(
                (
                    name,
                    fmt(values["si_sdr_input"], 2),
                    fmt(values["si_sdr_enhanced"], 2),
                    fmt(values["si_sdr_gain"], 2),
                    fmt(values["stoi_input"]),
                    fmt(values["stoi_enhanced"]),
                    fmt(values["stoi_gain"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 3. Safety and personalization observations",
            "",
            f"- Maximum observed mask magnitude: `{evaluation['mask_max']:.6f}`.",
            f"- Suppression-only mask invariant: `{evaluation['mask_invariant_passed']}`.",
            f"- Mean absolute output difference, Profile A vs B: `{evaluation['mean_output_abs_diff_profile_A_vs_B']:.8g}`.",
            f"- Mean absolute output difference, Profile B vs C: `{evaluation['mean_output_abs_diff_profile_B_vs_C']:.8g}`.",
            f"- Personalization signal threshold (`>1e-4`) passed: `{evaluation['personalization_signal_detected']}`.",
            "",
            "Phase A is an offline foundation model using `center=True` STFT/iSTFT. "
            "This report makes no causal streaming latency claim and reports no HASPI/HASQI result. "
            "Those are separate Phase B and deployment evaluation gates.",
            "",
            "## 4. Acceptance status",
            "",
            "This report is valid only when the checkpoint, manifest, per-sample CSV, "
            "and all recorded hashes remain available together. Phase B must not be "
            "started from this trainer until its HASPI surrogate and discriminator are "
            "implemented and independently validated.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Verified {checkpoint_path}, {manifest_path}, and {csv_path}")
    print(f"Generated {report_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_json", default="results/phase5_unet_film/phase5a_eval_results.json")
    parser.add_argument("--report", default="docs/phase_5a_verification_report.md")
    args = parser.parse_args()
    generate(ROOT / args.results_json, ROOT / args.report)


if __name__ == "__main__":
    main()
