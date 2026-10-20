# execution/phase5_unet_film/21_train_unet_film.py
# Phase A Training — Directive 05 Rev 2.1 §7
# Phase A: CompositeLoss only, 25 epochs, AdamW 3e-4, warmup 2 -> cosine, effective batch 16.
# Phase B is intentionally not implemented in this Phase A verification branch.
# Usage:
#   python 21_train_unet_film.py --phase A --data_dir data/processed --audiograms results/phase1_audiology/audiograms.json

import os, argparse, json, math, random, subprocess, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# Import model + dataset via importlib (handles numeric prefix)
import importlib.util as _ilu

def _load_module(fname):
    module_name = fname.replace(".py", "")
    spec = _ilu.spec_from_file_location(module_name, os.path.join(os.path.dirname(__file__), fname))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {fname}")
    mod = _ilu.module_from_spec(spec)
    # Register numeric-prefix modules before execution so Windows spawn can
    # pickle Dataset classes when DataLoader uses num_workers > 0.
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

_model_mod = _load_module("20_model_unet_film.py")
CausalUNetCRM = _model_mod.CausalUNetCRM
AudiogramEmbedding = _model_mod.AudiogramEmbedding

_data_mod = _load_module("03c_data_pipeline_v2.py")
HearingAidDatasetV2 = _data_mod.HearingAidDatasetV2
build_filelists = _data_mod.build_filelists
validate_filelists = _data_mod.validate_filelists

# ── 5 CompositeLoss (SI-SDR + MR-STFT + Compressed p=0.3 + Asymmetric + Gain + von Mises) ──
try:
    from asteroid.losses import pairwise_neg_sisdr
except Exception:
    pairwise_neg_sisdr = None

class CompositeLoss(nn.Module):
    def __init__(self, sr=16000, lambda_sisdr=1.0, lambda_mrstft=0.5,
                 lambda_comp=0.3, lambda_asym=0.5, lambda_gain=0.1, lambda_cons=0.05, p=0.3):
        super().__init__()
        self.sr = sr; self.p = p
        self.l_sisdr = lambda_sisdr; self.l_mrstft = lambda_mrstft
        self.l_comp = lambda_comp; self.l_asym = lambda_asym
        self.l_gain = lambda_gain; self.l_cons = lambda_cons
        self.mrstft_scales = [(256, 64), (512, 128), (1024, 256)]
        self.complex_eps = 1e-4

    def forward(self, est_wav, clean_wav, noisy_spec, m_real, m_imag, m_mag, clean_spec):
        if pairwise_neg_sisdr is not None:
            sisdr = pairwise_neg_sisdr(est_wav, clean_wav).mean()
        else:
            # SI-SDR fallback: -10 log10(||s_target||^2 / ||e_noise||^2)
            est = est_wav.squeeze(1); clean = clean_wav.squeeze(1)
            alpha = (est * clean).sum(1, keepdim=True) / (clean.pow(2).sum(1, keepdim=True) + 1e-8)
            s_target = alpha * clean
            e_noise = est - s_target
            sisdr_ratio = s_target.pow(2).sum(1) / (e_noise.pow(2).sum(1) + 1e-8)
            sisdr = (-10 * torch.log10(sisdr_ratio + 1e-8)).mean()

        mrstft = 0
        for n_fft, hop in self.mrstft_scales:
            win = torch.hann_window(n_fft, device=est_wav.device)
            s_est = torch.stft(est_wav.squeeze(1), n_fft=n_fft, hop_length=hop, window=win, center=True, return_complex=True)
            s_cln = torch.stft(clean_wav.squeeze(1), n_fft=n_fft, hop_length=hop, window=win, center=True, return_complex=True)
            mag_est, mag_cln = s_est.abs(), s_cln.abs()
            L = min(mag_est.shape[-1], mag_cln.shape[-1])
            mag_est, mag_cln = mag_est[..., :L], mag_cln[..., :L]
            sc = F.mse_loss(mag_est, mag_cln) / (mag_cln.mean() + 1e-7)
            mag = F.l1_loss(torch.log(mag_est + 1e-5), torch.log(mag_cln + 1e-5))
            mrstft += sc + mag
        mrstft /= len(self.mrstft_scales)

        enh_spec = torch.complex(m_real * noisy_spec.real - m_imag * noisy_spec.imag,
                                 m_real * noisy_spec.imag + m_imag * noisy_spec.real)
        # p=0.3 compressed complex. Use a magnitude floor and multiply by the
        # unit complex value directly: abs().pow(p) has an infinite derivative
        # at zero for p<1, which produces NaN gradients under bf16 autocast on
        # silent STFT bins.
        clean_mag = clean_spec.abs().clamp_min(self.complex_eps)
        enh_mag = enh_spec.abs().clamp_min(self.complex_eps)
        comp_clean = clean_spec * clean_mag.pow(self.p - 1.0)
        comp_enh = enh_spec * enh_mag.pow(self.p - 1.0)
        comp_clean_mag = clean_mag.pow(self.p)
        comp_enh_mag = enh_mag.pow(self.p)
        L_comp = F.mse_loss(comp_enh_mag, comp_clean_mag) + 0.5 * (
            F.mse_loss(comp_enh.real, comp_clean.real) + F.mse_loss(comp_enh.imag, comp_clean.imag)
        )

        speech_mask = (clean_mag > clean_mag.detach().max() * 10 ** (-40 / 20)).float()
        over = F.relu(0.3 - m_mag) * speech_mask
        L_asym = (over ** 2).mean() * 2.0
        L_gain = F.relu(m_mag - 1.0).pow(2).mean()
        # Stable cosine phase error: cos(dphase) = (enh.real*cln.real + enh.imag*cln.imag) / (enh_mag*cln_mag)
        # Avoids calling torch.angle() (atan2) which has singular derivatives at zero.
        cos_dphase = (enh_spec.real * clean_spec.real + enh_spec.imag * clean_spec.imag) / (enh_mag * clean_mag + 1e-7)
        L_cons = (1.0 - cos_dphase.clamp(-1.0, 1.0)).mean() * 0.05

        total = self.l_sisdr * sisdr + self.l_mrstft * mrstft + self.l_comp * L_comp + self.l_asym * L_asym + self.l_gain * L_gain + self.l_cons * L_cons
        logs = {"sisdr": sisdr.detach(), "mrstft": mrstft.detach(), "comp": L_comp.detach(),
                "asym": L_asym.detach(), "gain": L_gain.detach(), "cons": L_cons.detach()}
        return total, logs


# ── Helpers ─────────────────────────────────────────────────────────────────
def cosine_warmup(optimizer, epoch, total_epochs, warmup_epochs=2, base_lr=3e-4, min_lr=1e-6):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def git_state():
    try:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return {"commit": revision, "tracked_tree_clean": not bool(status), "tracked_status": status}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "tracked_tree_clean": False, "tracked_status": "unavailable"}


def run_metadata(args, device, model):
    git = git_state()
    return {
        "phase": "A",
        "args": vars(args).copy(),
        "git_commit": git["commit"],
        "git_tracked_tree_clean": git["tracked_tree_clean"],
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": str(device),
        "model_class": type(model).__name__,
        "model_params": model.count_params(),
    }

def build_dataloaders(args):
    clean, noise, rir = build_filelists(args.data_dir)
    try:
        counts = validate_filelists(
            clean, noise, rir,
            require_full=not getattr(args, "allow_incomplete_data", False),
        )
    except ValueError as exc:
        raise FileNotFoundError(f"{exc} Check --data_dir and the required VoiceBank/MUSAN/RIR layout.") from exc
    print(f"[Data] clean={counts['clean']} noise={counts['noise']} rir={counts['rir']}")
    full = HearingAidDatasetV2(clean, noise, rir, args.audiograms, segment=args.segment, sr=16000)
    n_val = max(1, int(len(full) * 0.05))
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True, prefetch_factor=2 if args.num_workers > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader

# ── Phase A ──────────────────────────────────────────────────────────────────
def train_phase_a(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    if device.type == "cpu" and args.mixed_precision != "no":
        raise RuntimeError("bf16/fp16 Phase A training requires CUDA; use a local NVIDIA GPU or pass --mixed_precision no for a smoke test")
    print(f"[Phase A] device={device} cuda={torch.cuda.is_available()} torch={torch.__version__}")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    train_loader, val_loader = build_dataloaders(args)
    model = CausalUNetCRM(n_fft=args.n_fft).to(device)
    print(f"[Phase A] params {model.count_params():,} ({model.count_params()/1e6:.3f}M)")
    metadata = run_metadata(args, device, model)
    print(f"[Phase A] git={metadata['git_commit']}")
    if not metadata["git_tracked_tree_clean"]:
        raise RuntimeError(
            "Tracked Git changes are present; commit the Phase 5A source before training "
            "so checkpoint provenance identifies the exact code."
        )
    criterion = CompositeLoss(p=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))
    use_scaler = (args.mixed_precision == "fp16")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    use_amp = (args.mixed_precision in ("bf16", "fp16"))

    start_epoch = 0
    best_val = float("inf")
    patience = 5
    patience_ctr = 0
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_val = float(checkpoint.get("best_val", checkpoint.get("val_loss", best_val)))
        patience_ctr = int(checkpoint.get("patience_ctr", 0))
        print(f"[Phase A] resumed epoch={start_epoch} best_val={best_val:.4f} from {args.resume}")

    for epoch in range(start_epoch, args.epochs):
        lr = cosine_warmup(optimizer, epoch, args.epochs, warmup_epochs=2, base_lr=args.lr)
        model.train()
        running = 0
        for step, batch in enumerate(train_loader):
            noisy = batch["noisy"].to(device)   # (B,1,64000)
            clean = batch["clean"].to(device)
            audiogram = batch["audiogram"].to(device)  # (B,6)

            # Specs needed for loss
            window = model.window
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                enh, (m_real, m_imag, m_mag, m_phase), _ = model(noisy, audiogram)
                # Specs for loss — center=True required for Hann COLA (training; streaming harness keeps causal buffer separately)
                clean_spec = torch.stft(clean.squeeze(1), n_fft=args.n_fft, hop_length=64, window=window, center=True, return_complex=True)
                noisy_spec = torch.stft(noisy.squeeze(1), n_fft=args.n_fft, hop_length=64, window=window, center=True, return_complex=True)
                loss, logs = criterion(enh, clean, noisy_spec, m_real, m_imag, m_mag, clean_spec)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite Phase A loss at epoch={epoch + 1} step={step}")
                loss = loss / args.accum
                if step % 100 == 0:
                    print(f"  step {step}/{len(train_loader)} loss {float(loss * args.accum):.4f}", flush=True)

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                print(f"  [Warning] non-finite Phase A gradient at epoch={epoch + 1} step={step}, skipping step", flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue
            if (step + 1) % args.accum == 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if use_scaler:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach() * args.accum)

        # Handle leftover grads
        if len(train_loader) % args.accum != 0:
            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if use_scaler:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                noisy = batch["noisy"].to(device); clean = batch["clean"].to(device); audiogram = batch["audiogram"].to(device)
                enh, (m_real, m_imag, m_mag, m_phase), _ = model(noisy, audiogram)
                clean_spec = torch.stft(clean.squeeze(1), n_fft=args.n_fft, hop_length=64, window=model.window, center=True, return_complex=True)
                noisy_spec = torch.stft(noisy.squeeze(1), n_fft=args.n_fft, hop_length=64, window=model.window, center=True, return_complex=True)
                loss, _ = criterion(enh, clean, noisy_spec, m_real, m_imag, m_mag, clean_spec)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite Phase A validation loss at epoch={epoch + 1}")
                val_loss += float(loss)
        val_loss /= max(1, len(val_loader))
        train_loss = running / max(1, len(train_loader))
        print(f"[Phase A] epoch {epoch+1}/{args.epochs} lr {lr:.2e} train {train_loss:.3f} val {val_loss:.3f}", flush=True)

        # Save best and last
        should_stop = False
        if val_loss < best_val:
            best_val = val_loss; patience_ctr = 0
            torch.save({"epoch": epoch+1, "model": model.state_dict(), "val_loss": val_loss, "best_val": best_val, "metadata": metadata},
                       os.path.join(args.ckpt_dir, "best_composite.pt"))
            print(f"  -> new best_composite.pt (val {val_loss:.3f})", flush=True)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  -> early stop patience {patience} at epoch {epoch+1}", flush=True)
                should_stop = True
        torch.save({"epoch": epoch+1, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "val_loss": val_loss, "best_val": best_val, "patience_ctr": patience_ctr, "metadata": metadata},
                   os.path.join(args.ckpt_dir, "last.pt"))
        if should_stop:
            break

# Phase B is deliberately not exposed here. Its HASPI surrogate and
# MetricGAN+/LSGAN discriminator require separate, validated implementations.


def main():
    p = argparse.ArgumentParser(description="Phase 5A U-Net training (Rev 2.1)")
    p.add_argument("--phase", type=str, default="A", choices=["A"], help="Phase A composite foundation training")
    p.add_argument("--data_dir", type=str, default="data/processed")
    p.add_argument("--audiograms", type=str, default="results/phase1_audiology/audiograms.json")
    p.add_argument("--ckpt_dir", type=str, default="results/phase5_unet_film/checkpoints")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n_fft", type=int, default=256)
    p.add_argument("--segment", type=int, default=64000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16", "no"])
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Training device; auto selects CUDA when available")
    p.add_argument("--resume", type=str, default=None, help="Resume Phase A from a last.pt checkpoint")
    p.add_argument("--allow_incomplete_data", action="store_true", help="Permit non-canonical file counts for smoke tests only")
    args = p.parse_args()
    if args.epochs is None:
        args.epochs = 25
    train_phase_a(args)

if __name__ == "__main__":
    main()
