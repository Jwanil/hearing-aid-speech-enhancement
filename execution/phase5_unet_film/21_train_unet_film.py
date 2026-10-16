# execution/phase5_unet_film/21_train_unet_film.py
# Two-Phase Training — Directive 05 Rev 2.1 §7
# Phase A: CompositeLoss only, 25 epochs, AdamW 3e-4, warmup 2 -> cosine, effective batch 16.
# Phase B: Pilots B1 (Frozen HASPINet, 5ep 2.5h) / B2 (LSGAN+VSN, 5ep 3.5h) gated on ΔHASPI >=0.015.
# Usage:
#   python 21_train_unet_film.py --phase A --data_dir data/processed --audiograms results/phase1_audiology/audiograms.json
#   python 21_train_unet_film.py --phase B1 --init_ckpt results/phase5_unet_film/checkpoints/best_composite.pt
#   python 21_train_unet_film.py --phase B2 --init_ckpt results/phase5_unet_film/checkpoints/best_composite.pt

import os, argparse, json, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# Import model + dataset via importlib (handles numeric prefix)
import importlib.util as _ilu

def _load_module(fname):
    spec = _ilu.spec_from_file_location(fname.replace(".py",""), os.path.join(os.path.dirname(__file__), fname))
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_model_mod = _load_module("20_model_unet_film.py")
CausalUNetCRM = _model_mod.CausalUNetCRM
AudiogramEmbedding = _model_mod.AudiogramEmbedding

_data_mod = _load_module("03c_data_pipeline_v2.py")
HearingAidDatasetV2 = _data_mod.HearingAidDatasetV2
build_filelists = _data_mod.build_filelists

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
        # p=0.3 compressed complex: |.|^p * exp(j angle)
        comp_clean = clean_spec.abs().pow(self.p) * torch.exp(1j * clean_spec.angle())
        comp_enh = enh_spec.abs().pow(self.p) * torch.exp(1j * enh_spec.angle())
        L_comp = F.mse_loss(comp_enh.abs(), comp_clean.abs()) + 0.5 * (
            F.mse_loss(comp_enh.real, comp_clean.real) + F.mse_loss(comp_enh.imag, comp_clean.imag)
        )

        speech_mask = (clean_spec.abs() > clean_spec.abs().max() * 10 ** (-40 / 20)).float()
        over = F.relu(0.3 - m_mag) * speech_mask
        L_asym = (over ** 2).mean() * 2.0
        L_gain = F.relu(m_mag - 1.0).pow(2).mean()
        dphase = enh_spec.angle() - clean_spec.angle()
        L_cons = (1 - torch.cos(dphase)).mean() * 0.05

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

def build_dataloaders(args):
    clean, noise, rir = build_filelists(args.data_dir)
    if not clean:
        raise FileNotFoundError(f"No clean files found under {args.data_dir}. Check --data_dir and globs.")
    if not noise:
        print(f"WARNING: no noise files found under {args.data_dir} — training without additive noise")
    full = HearingAidDatasetV2(clean, noise, rir, args.audiograms, segment=args.segment, sr=16000)
    n_val = max(1, int(len(full) * 0.05))
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True, prefetch_factor=2 if args.num_workers > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader

# ── Phase A ──────────────────────────────────────────────────────────────────
def train_phase_a(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase A] device={device} cuda={torch.cuda.is_available()} torch={torch.__version__}")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    train_loader, val_loader = build_dataloaders(args)
    model = CausalUNetCRM(n_fft=args.n_fft).to(device)
    print(f"[Phase A] params {model.count_params():,} ({model.count_params()/1e6:.3f}M)")
    criterion = CompositeLoss(p=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))
    use_scaler = (args.mixed_precision == "fp16")
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    use_amp = (args.mixed_precision in ("bf16", "fp16"))

    best_val = float("inf")
    patience = 5
    patience_ctr = 0

    for epoch in range(args.epochs):
        lr = cosine_warmup(optimizer, epoch, args.epochs, warmup_epochs=2, base_lr=args.lr)
        model.train()
        running = 0
        for step, batch in enumerate(train_loader):
            noisy = batch["noisy"].to(device)   # (B,1,64000)
            clean = batch["clean"].to(device)
            audiogram = batch["audiogram"].to(device)  # (B,6)

            # Specs needed for loss
            window = model.window
            with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                enh, (m_real, m_imag, m_mag, m_phase), _ = model(noisy, audiogram)
                # Specs for loss — center=True required for Hann COLA (training; streaming harness keeps causal buffer separately)
                clean_spec = torch.stft(clean.squeeze(1), n_fft=args.n_fft, hop_length=64, window=window, center=True, return_complex=True)
                noisy_spec = torch.stft(noisy.squeeze(1), n_fft=args.n_fft, hop_length=64, window=window, center=True, return_complex=True)
                loss, logs = criterion(enh, clean, noisy_spec, m_real, m_imag, m_mag, clean_spec)
                loss = loss / args.accum
                if step % 100 == 0:
                    print(f"  step {step}/{len(train_loader)} loss {float(loss * args.accum):.4f}")

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
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
                val_loss += float(loss)
        val_loss /= max(1, len(val_loader))
        train_loss = running / max(1, len(train_loader))
        print(f"[Phase A] epoch {epoch+1}/{args.epochs} lr {lr:.2e} train {train_loss:.3f} val {val_loss:.3f}")

        # Save best and last
        torch.save({"epoch": epoch+1, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "val_loss": val_loss},
                   os.path.join(args.ckpt_dir, "last.pt"))
        if val_loss < best_val:
            best_val = val_loss; patience_ctr = 0
            torch.save({"epoch": epoch+1, "model": model.state_dict(), "val_loss": val_loss},
                       os.path.join(args.ckpt_dir, "best_composite.pt"))
            print(f"  -> new best_composite.pt (val {val_loss:.3f})")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  -> early stop patience {patience} at epoch {epoch+1}")
                break

# ── Phase B pilots — SKELETON (Phase A only ships; B1/B2 gate TODO per §7.2) ─
# Do not claim B1/B2 as implemented until HASPINet + HASPIDiscriminator + LSGAN+R1 are present.
def train_phase_b(args):
    # B1/B2 share same entry — B1 is stable path; B2 requires HASPIDiscriminator training.
    # Full B2 is gated; this skeleton runs B1 (composite + frozen haspi surrogate) and a passthrough B2.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase {args.phase}] device={device} init_ckpt={args.init_ckpt}")
    ckpt = torch.load(args.init_ckpt, map_location=device)
    model = CausalUNetCRM(n_fft=args.n_fft).to(device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    print(f"[Phase {args.phase}] loaded {args.init_ckpt} params {model.count_params():,}")
    print(f"[Phase {args.phase}] — pilot runs from best_composite.pt. Implement haspi_net/D per directive §7.2 for full HASPI gate.")
    print(f"[Phase {args.phase}] Skeleton: fine-tuning with reduced lr {args.lr_b} via CompositeLoss only (replace with L_comp+0.05*L_haspi for B1).")
    train_loader, val_loader = build_dataloaders(args)
    criterion = CompositeLoss(p=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_b, weight_decay=1e-4)
    use_scaler_b = (args.mixed_precision == "fp16")
    amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler_b)
    use_amp_b = (args.mixed_precision in ("bf16", "fp16"))
    os.makedirs(args.ckpt_dir, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            noisy = batch["noisy"].to(device); clean = batch["clean"].to(device); audiogram = batch["audiogram"].to(device)
            with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp_b):
                enh, (m_real, m_imag, m_mag, m_phase), _ = model(noisy, audiogram)
                clean_spec = torch.stft(clean.squeeze(1), n_fft=args.n_fft, hop_length=64, window=model.window, center=True, return_complex=True)
                noisy_spec = torch.stft(noisy.squeeze(1), n_fft=args.n_fft, hop_length=64, window=model.window, center=True, return_complex=True)
                loss, _ = criterion(enh, clean, noisy_spec, m_real, m_imag, m_mag, clean_spec)
                # B1: loss = loss + 0.05 * (1 - haspi_net(enh, audiogram).mean())
                # B2: + LSGAN term via HASPIDiscriminator
                loss = loss / args.accum
            if use_scaler_b: scaler.scale(loss).backward()
            else: loss.backward()
            if use_scaler_b: scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if use_scaler_b: scaler.step(optimizer); scaler.update()
            else: optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        print(f"[Phase {args.phase}] epoch {epoch+1}/{args.epochs} done — ckpt to {args.ckpt_dir}")
        torch.save({"epoch": epoch+1, "model": model.state_dict()}, os.path.join(args.ckpt_dir, f"{args.phase}_epoch{epoch+1:02d}.pt"))
    torch.save({"model": model.state_dict()}, os.path.join(args.ckpt_dir, f"best_{args.phase.lower()}.pt"))
    print(f"[Phase {args.phase}] saved best_{args.phase.lower()}.pt — eval gate: ΔHASPI >=0.015 and HASQI >= -0.01 on val n=200 Profile B")


def main():
    p = argparse.ArgumentParser(description="Phase 5 U-Net Two-Phase Training (Rev 2.1)")
    p.add_argument("--phase", type=str, default="A", choices=["A", "B1", "B2"], help="A=composite 25ep, B1=frozen haspi 5ep, B2=LSGAN 5ep")
    p.add_argument("--data_dir", type=str, default="data/processed")
    p.add_argument("--audiograms", type=str, default="results/phase1_audiology/audiograms.json")
    p.add_argument("--ckpt_dir", type=str, default="results/phase5_unet_film/checkpoints")
    p.add_argument("--init_ckpt", type=str, default=None, help="Required for B1/B2")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_b", type=float, default=1e-4)
    p.add_argument("--n_fft", type=int, default=256)
    p.add_argument("--segment", type=int, default=64000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16", "no"])
    args = p.parse_args()
    if args.epochs is None:
        args.epochs = 25 if args.phase == "A" else 5
    if args.phase in ("B1", "B2") and not args.init_ckpt:
        p.error("--init_ckpt is required for B1/B2 (path to best_composite.pt)")
    if args.phase == "A":
        train_phase_a(args)
    else:
        train_phase_b(args)

if __name__ == "__main__":
    main()
