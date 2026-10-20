# RTX 4090 Workstation Prompt — Corrected Phase 5A Retraining

You are continuing the Audiogram-Personalized Speech Enhancement project on the
physical Windows RTX 4090 workstation. Work from the repository root and do not
claim success until every gate below has passed.

## Scope

This run is **Phase 5A only**:

- Train `CausalUNetCRM` with the composite loss for 25 epochs.
- Verify the bf16 numerical-stability fixes.
- Produce a reproducible held-out VoiceBank-DEMAND evaluation.
- Record checkpoint, dataset, manifest, environment, and Git provenance.

Do **not** run Phase B1 or B2. They are intentionally disabled in this branch
because a validated pretrained HASPI surrogate and a real MetricGAN+/LSGAN
discriminator are not implemented yet.

## What was wrong in the previous Phase 5A package

1. The old evaluator used `build_filelists("data/processed")`, which points to
   the dynamic training clean corpus. Its first 100 samples were not a fixed
   held-out test set.
2. Evaluation randomly selected crops, noise, SNR, RIR, and audiograms without
   recording those choices, so the aggregate numbers were not exactly
   reproducible.
3. The old result JSON did not include a checkpoint hash, dataset manifest,
   Git revision, environment versions, or per-sample metrics.
4. The old report described the profile difference as RMS although the code
   computed mean absolute waveform difference.
5. The old report presented the Phase A model as causal/streaming and reported
   latency even though the trained model uses `center=True` STFT/iSTFT. Phase A
   is an offline foundation model; do not make a causal latency claim from it.
6. The old trainer exposed incomplete B1/B2 code. The corrected trainer accepts
   Phase A only and refuses to pretend that Phase B is implemented.

The corrected evaluator now uses the official non-overlapping VoiceBank test
pair directories and writes a manifest, per-sample CSV, aggregate JSON, and
SHA-256 provenance fields.

## 1. Pull and inspect the corrected code

```powershell
git pull --ff-only origin main
git rev-parse --short HEAD
git status --short
```

Use the exact correction commit supplied with this prompt. If the commit is not
available from `origin`, stop and push/transfer it before starting the run:

```powershell
$requiredCommit = '<PHASE5A_CORRECTION_COMMIT>'
if ($requiredCommit -eq '<PHASE5A_CORRECTION_COMMIT>') { throw 'Set requiredCommit to the pushed correction commit' }
$currentShort = (git rev-parse --short=7 HEAD).Trim()
$expectedShort = $requiredCommit.Substring(0, [Math]::Min(7, $requiredCommit.Length))
if ($currentShort -ne $expectedShort) {
    throw "Wrong correction commit. Expected $requiredCommit"
}
if ((git status --porcelain --untracked-files=no).Trim()) {
    throw 'Tracked Git changes are present; do not train from a dirty source tree'
}
```

The checkout must contain:

```text
execution/phase5_unet_film/03c_data_pipeline_v2.py
execution/phase5_unet_film/20_model_unet_film.py
execution/phase5_unet_film/21_train_unet_film.py
execution/phase5_unet_film/24_verify_local_training.py
execution/phase5_unet_film/26_eval_phase5a.py
execution/phase5_unet_film/27_generate_phase5a_report.py
execution/phase5_unet_film/28_verify_phase5a_protocol.py
phase5a_retraining_prompt.md
```

Do not overwrite unrelated local work. Do not add the dataset or checkpoints
to Git.

## 2. Verify Python, CUDA, and the GPU

Activate the existing project environment if it exists. A matched CUDA
PyTorch/torchaudio pair is required; do not use a CPU-only wheel.

```powershell
python --version
python -c "import torch, torchaudio; print('torch', torch.__version__); print('torchaudio', torchaudio.__version__); print('cuda build', torch.version.cuda); print('cuda available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)"
nvidia-smi
```

Required:

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
bf16: True
```

Keep the already-installed matched CUDA pair if it passes. Only reinstall if
the verification fails, and install matching versions together, for example:

```powershell
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install soundfile numpy scipy pywavelets matplotlib pystoi
```

`pesq` is optional for this Phase 5A gate and is not required to train or run
the corrected SI-SDR/STOI evaluator.

## 3. Locate the SSD dataset safely

The SSD must remain connected to this workstation. First identify the actual
drive letter and dataset root; do not assume `E:\Minor Project\Data`.

```powershell
Get-Volume | Where-Object DriveType -eq 'Removable' | Select-Object DriveLetter,FileSystemLabel,Size
```

Enter the directory that directly contains `processed`:

```powershell
$targetPath = Read-Host 'Enter the SSD dataset root that directly contains processed'
if ([string]::IsNullOrWhiteSpace($targetPath)) { throw 'Dataset root is required' }
$targetPath = (Resolve-Path $targetPath -ErrorAction Stop).Path
Test-Path "$targetPath\processed\voicebank\clean_trainset_28spk_wav"
Test-Path "$targetPath\processed\musan"
Test-Path "$targetPath\processed\rir"
Test-Path "$targetPath\processed\voicebank\clean_testset_wav"
Test-Path "$targetPath\processed\voicebank\noisy_testset_wav"
```

All five checks must return `True`.

Create the repository junction only after confirming the target. Never remove a
real `data` directory automatically:

```powershell
if (Test-Path data) {
    $existing = Get-Item data -Force
    if ($existing.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        cmd /c rmdir data
    } else {
        throw 'data exists and is not a junction; inspect it manually before continuing'
    }
}
cmd /c mklink /J data "$targetPath"
Get-Item data -Force
```

## 4. Run the code and protocol checks

```powershell
python -m py_compile `
  execution/phase5_unet_film/03c_data_pipeline_v2.py `
  execution/phase5_unet_film/20_model_unet_film.py `
  execution/phase5_unet_film/21_train_unet_film.py `
  execution/phase5_unet_film/22_streaming_inference.py `
  execution/phase5_unet_film/24_verify_local_training.py `
  execution/phase5_unet_film/26_eval_phase5a.py `
  execution/phase5_unet_film/27_generate_phase5a_report.py `
  execution/phase5_unet_film/28_verify_phase5a_protocol.py

python execution/phase5_unet_film/28_verify_phase5a_protocol.py
python execution/phase5_unet_film/20_model_unet_film.py
```

Expected protocol result:

```text
PHASE5A EVALUATION PROTOCOL PASS
```

## 5. Run the strict dataset and CUDA gate

```powershell
python -c "import importlib.util; s=importlib.util.spec_from_file_location('d','execution/phase5_unet_film/03c_data_pipeline_v2.py'); d=importlib.util.module_from_spec(s); s.loader.exec_module(d); c,n,r=d.build_filelists('data/processed'); print(d.validate_filelists(c,n,r))"
python execution/phase5_unet_film/24_verify_local_training.py
```

Required file-count result:

```text
{'clean': 11071, 'noise': 930, 'rir': 60000}
```

Required CUDA gate result:

```text
DATASET AND CUDA BF16 ONE-STEP PASS (...)
```

Do not use `--allow_incomplete_data` for this run. Do not substitute NOIZEUS
`clean/` or `noisy/` for VoiceBank training data.

## 6. Archive old checkpoints, then retrain Phase A from scratch

Do not delete the old checkpoint until it has been archived. If the directory
exists, rename it to a dated legacy directory:

```powershell
if (Test-Path results/phase5_unet_film/checkpoints) {
    $legacyName = "checkpoints_legacy_{0}" -f (Get-Date -Format 'yyyyMMdd_HHmmss')
    if (Test-Path "results/phase5_unet_film/$legacyName") {
        throw "Legacy checkpoint destination already exists: $legacyName"
    }
    Rename-Item results/phase5_unet_film/checkpoints $legacyName -ErrorAction Stop
}
New-Item -ItemType Directory -Force results/phase5_unet_film/checkpoints | Out-Null
if ((Get-ChildItem results/phase5_unet_film/checkpoints -Force | Measure-Object).Count -ne 0) {
    throw 'New checkpoint directory is not empty'
}
```

Start a fresh 25-epoch Phase A run:

```powershell
python execution/phase5_unet_film/21_train_unet_film.py `
  --phase A `
  --device cuda `
  --mixed_precision bf16 `
  --data_dir data/processed `
  --audiograms results/phase1_audiology/audiograms.json `
  --ckpt_dir results/phase5_unet_film/checkpoints `
  --batch 8 `
  --accum 2 `
  --num_workers 2
```

Use `--num_workers 4` only if the first run is stable. The run must show finite
losses and must create:

```text
results/phase5_unet_film/checkpoints/best_composite.pt
results/phase5_unet_film/checkpoints/last.pt
```

Check progress from a second terminal if needed:

```powershell
python execution/phase5_unet_film/check_progress.py
```

## 7. Run the corrected held-out evaluation

This is the accepted Phase 5A evaluation. It uses the official test split,
not dynamic training augmentation. Run all available test pairs:

```powershell
python execution/phase5_unet_film/26_eval_phase5a.py `
  --data_dir data/processed `
  --checkpoint results/phase5_unet_film/checkpoints/best_composite.pt `
  --results_json results/phase5_unet_film/phase5a_eval_results.json `
  --results_csv results/phase5_unet_film/phase5a_eval_per_sample.csv `
  --manifest results/phase5_unet_film/phase5a_eval_manifest.csv `
  --seed 0
```

Verify that the resulting JSON contains:

```text
protocol = official_voicebank_test_pairs
dataset.evaluation_split = VoiceBank-DEMAND official clean_testset_wav/noisy_testset_wav
dataset.pair_count = 824  (for the canonical VoiceBank test release)
evaluation.mask_invariant_passed = true
artifacts.per_sample_csv exists
  artifacts.per_sample_csv_sha256 matches the file
  checkpoint.sha256 is present
  git_commit is present
  docs/phase_5a_verification_report.md was generated from that JSON
```

Do not report the historical `+8.14 dB` result unless it is reproduced by this
corrected evaluator. Do not report `1.82 ms/frame` as a causal latency result;
Phase A currently uses the offline `center=True` path.

## 8. Regenerate and inspect the report

Generate `docs/phase_5a_verification_report.md` directly from the JSON. Do not
manually copy metric values into the report:

```powershell
python execution/phase5_unet_film/27_generate_phase5a_report.py `
  --results_json results/phase5_unet_film/phase5a_eval_results.json `
  --report docs/phase_5a_verification_report.md
```

The generated report must state:

- the exact test pair count;
- the checkpoint hash and Git revision;
- the exact evaluator command;
- SI-SDR/STOI aggregate and per-profile values;
- the suppression-only mask result;
- that Phase A is an offline foundation model;
- that HASPI/HASQI and true personalization remain future Phase B gates.

Do not include unsupported claims about causal streaming, acoustic feedback
elimination, human clinical benefit, or Phase B completion.

## 9. Final artifact checklist

```text
[ ] CUDA/bf16 gate passed on RTX 4090
[ ] Dataset gate passed 11071/930/60000
[ ] Protocol smoke passed
[ ] 25 Phase A epochs completed with finite loss
[ ] best_composite.pt and last.pt exist locally but are not Git-tracked
[ ] Official test pair manifest written
[ ] Per-sample CSV written
[ ] Aggregate JSON includes hashes and environment metadata
[ ] Corrected report was generated from the JSON and matches the hashes
[ ] No Phase B command was run
[ ] No dataset, WAV, PT, or PTH file was added to Git
```

Only after all checks pass should the new Phase 5A result be considered valid.
