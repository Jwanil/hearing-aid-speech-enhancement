# Training Platform — Local RTX 4090 (Canonical)

## Decision

Phase 5 U-Net/FiLM and Phase 5b Mamba training run on the team's **local NVIDIA
RTX 4090 workstation**. Cloud GPU allocation and migration are not part of the
current training plan; the local workstation is the sole canonical platform.

The local machine provides:

- No hourly GPU cost or cloud-balance risk.
- Persistent local dataset and checkpoints.
- Full control over CUDA, PyTorch, worker count, and process lifetime.
- Reproducible restarts using `last.pt` without re-uploading data.

## Required local environment

| Component | Requirement |
|---|---|
| GPU | NVIDIA RTX 4090, 24 GB VRAM |
| Driver | Current NVIDIA driver supporting the installed CUDA runtime |
| Python | 3.11 or 3.12 recommended |
| PyTorch | CUDA-enabled build; `torch.version.cuda` must not be `None` |
| Precision | `bf16` for Phase A; RTX 4090 supports bfloat16 |
| Storage | At least 25 GB free for the verified corpus, temporary files, and checkpoints |
| Checkpoints | `results/phase5_unet_film/checkpoints/` on persistent local storage |

Do not use a CPU-only PyTorch wheel on the 4090 machine. Verify the actual
environment before training:

```bash
nvidia-smi

python - <<'PY'
import torch, torchaudio

print("torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available()
print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("bf16:", torch.cuda.is_bf16_supported())
print("torchaudio:", torchaudio.__version__)
PY
```

## Canonical dataset layout

The Phase A loader requires the complete corpus below. These are local files,
not Git-tracked artifacts:

```text
data/processed/
├── voicebank/clean_trainset_28spk_wav/   11,071 WAV files
├── musan/                                    930 WAV files
└── rir/                                   60,000 WAV files
```

NOIZEUS `clean/` and `noisy/` directories are evaluation-only. They must not
be used as VoiceBank training substitutes.

Run the strict gate from the repository root before every new training run:

```bash
python - <<'PY'
import importlib.util

path = "execution/phase5_unet_film/03c_data_pipeline_v2.py"
spec = importlib.util.spec_from_file_location("phase5_data", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

clean, noise, rir = mod.build_filelists("data/processed")
counts = mod.validate_filelists(clean, noise, rir)
print(counts)
assert counts == {"clean": 11071, "noise": 930, "rir": 60000}
print("DATASET FILELIST PASS")
PY
```

The loader also uses FFT-based RIR augmentation and excludes macOS `._*`
resource-fork files. Do not delete or replace a corpus directory unless a
verified backup/archive is available.

## Install and checks

Install a CUDA-enabled PyTorch build appropriate for the local driver first,
then install the project dependencies. The exact PyTorch command depends on
the machine's supported CUDA runtime; verify it with the command above.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the CUDA-enabled torch/torchaudio build selected for this host.
python -m pip install -r requirements.txt
```

Run the focused Phase 5 checks:

```bash
python -m py_compile \
  execution/phase5_unet_film/03c_data_pipeline_v2.py \
  execution/phase5_unet_film/20_model_unet_film.py \
  execution/phase5_unet_film/21_train_unet_film.py \
  execution/phase5_unet_film/22_streaming_inference.py

python execution/phase5_unet_film/20_model_unet_film.py
```

Before a full run, perform a CUDA bf16 one-step smoke test using a real batch.
Require finite forward output, loss, gradients, and parameters. The training
script now fails fast if any loss or gradient becomes non-finite.

The repository provides the strict gate directly:

```bash
python execution/phase5_unet_film/24_verify_local_training.py
```

## Phase A command

Run from the repository root and keep checkpoints on persistent local storage:

```bash
python execution/phase5_unet_film/21_train_unet_film.py \
  --phase A \
  --device cuda \
  --data_dir data/processed \
  --audiograms results/phase1_audiology/audiograms.json \
  --ckpt_dir results/phase5_unet_film/checkpoints \
  --batch 8 \
  --accum 2 \
  --num_workers 4 \
  --mixed_precision bf16
```

Expected effective batch size is `16`. The run writes:

```text
results/phase5_unet_film/checkpoints/last.pt
results/phase5_unet_film/checkpoints/best_composite.pt
```

To resume after an interruption:

```bash
python execution/phase5_unet_film/21_train_unet_film.py \
  --phase A --device cuda --mixed_precision bf16 \
  --data_dir data/processed \
  --audiograms results/phase1_audiology/audiograms.json \
  --ckpt_dir results/phase5_unet_film/checkpoints \
  --resume results/phase5_unet_film/checkpoints/last.pt
```

The `--allow_incomplete_data` option exists only for deliberate smoke tests;
never use it for the canonical Phase A experiment.

## Platform policy

- **Local RTX 4090:** canonical Phase 5/5b training platform.
- **CPU:** smoke tests only, never a Phase A training run.
- **Cloud GPUs:** not used for the current training plan.

The source of truth for the executable behavior is the Phase 5 scripts in
`execution/phase5_unet_film/`; this document describes the local platform and
verification gate around them.
