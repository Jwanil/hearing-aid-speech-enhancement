"""Strict local RTX 4090/data/one-step gate for Phase 5 training.

Run from the repository root before Phase A. This script does not write a
checkpoint or modify the dataset.
"""

import argparse
import importlib.util
import os
from types import SimpleNamespace

import torch


HERE = os.path.dirname(__file__)


def load_module(filename):
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), os.path.join(HERE, filename))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="Verify local Phase 5 CUDA/data/training readiness")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--audiograms", default="results/phase1_audiology/audiograms.json")
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this gate on the local RTX 4090 workstation")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The active CUDA device does not report bfloat16 support")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__} CUDA build: {torch.version.cuda}")

    train = load_module("21_train_unet_film.py")
    data_args = SimpleNamespace(
        data_dir=args.data_dir,
        audiograms=args.audiograms,
        segment=64000,
        batch=args.batch,
        num_workers=0,
        allow_incomplete_data=False,
    )
    train_loader, _ = train.build_dataloaders(data_args)
    batch = next(iter(train_loader))
    device = torch.device("cuda")
    noisy = batch["noisy"].to(device)
    clean = batch["clean"].to(device)
    audiogram = batch["audiogram"].to(device)

    model = train.CausalUNetCRM(n_fft=256).to(device)
    criterion = train.CompositeLoss(p=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        enhanced, (m_real, m_imag, m_mag, _), _ = model(noisy, audiogram)
        clean_spec = torch.stft(clean.squeeze(1), n_fft=256, hop_length=64,
                                window=model.window, center=True, return_complex=True)
        noisy_spec = torch.stft(noisy.squeeze(1), n_fft=256, hop_length=64,
                                window=model.window, center=True, return_complex=True)
        loss, _ = criterion(enhanced, clean, noisy_spec, m_real, m_imag, m_mag, clean_spec)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite smoke loss: {loss}")
    loss.backward()
    bad_gradients = [name for name, parameter in model.named_parameters()
                     if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
    if bad_gradients:
        raise FloatingPointError(f"non-finite smoke gradients: {bad_gradients[:5]}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    print(f"DATASET AND CUDA BF16 ONE-STEP PASS (loss={float(loss):.5f})")


if __name__ == "__main__":
    main()
