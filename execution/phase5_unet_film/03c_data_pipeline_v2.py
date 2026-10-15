# execution/phase5_unet_film/03c_data_pipeline_v2.py
# HearingAidDatasetV2 — Directive 05 Rev 2.1 §6
# Dynamic SNR mix -5..20 dB, RIR 50% (openSLR26 RT60 0.2-0.6s, 8000 taps), stratified audiograms B/C 40%.
# Single source of truth — Phase 5b imports this file.

import os, json, random
import torch
import torchaudio


class HearingAidDatasetV2(torch.utils.data.Dataset):
    def __init__(self, clean_files, noise_files, rir_files, audiograms_json, segment=64000, sr=16000):
        self.clean = clean_files if isinstance(clean_files, list) else list(clean_files)
        self.noise = noise_files if isinstance(noise_files, list) else list(noise_files)
        self.rir = rir_files if isinstance(rir_files, list) else (list(rir_files) if rir_files else [])
        # audiograms_json may be dict path or already-loaded dict
        if isinstance(audiograms_json, dict):
            ag_data = audiograms_json
        elif isinstance(audiograms_json, str) and os.path.exists(audiograms_json):
            with open(audiograms_json) as f:
                ag_data = json.load(f)
            # Normalize various json shapes: {name: [6]} or {test_profiles: [...]} or {random_audiograms: [...]}
            if isinstance(ag_data, dict) and any(k in ag_data for k in ("test_profiles", "random_audiograms")):
                audiograms = {}
                for p in ag_data.get("test_profiles", []):
                    audiograms[p["name"]] = p["levels"]
                for i, p in enumerate(ag_data.get("random_audiograms", [])):
                    audiograms[f"random_{i}"] = p["levels"]
                ag_data = audiograms
        else:
            # Fallback: three canonical profiles for stratified training
            ag_data = {
                "A_normal": [10, 10, 10, 10, 10, 10],
                "B_presbycusis": [10, 15, 20, 45, 70, 85],
                "C_flat_severe": [60, 60, 60, 60, 60, 60],
            }
            # Add 20 synthetic random audiograms via cumsum jitter
            import numpy as np
            rng_np = np.random.default_rng(0)
            for i in range(20):
                base = int(rng_np.integers(10, 50))
                jitter = np.cumsum(rng_np.normal(0, 8, 6))
                jitter = np.clip(base + jitter, 0, 90).astype(int).tolist()
                ag_data[f"random_{i}"] = jitter
        self.audiograms = ag_data
        self.keys = list(self.audiograms.keys())
        self.segment = segment
        self.sr = sr

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        clean, _ = torchaudio.load(self.clean[idx])
        clean = clean.mean(0, keepdim=True) if clean.shape[0] > 1 else clean
        clean = self._crop_pad(clean)

        ag_name = random.choice(self.keys)
        ag = torch.tensor(self.audiograms[ag_name], dtype=torch.float32) / 120.0

        n_path = random.choice(self.noise)
        noise, _ = torchaudio.load(n_path)
        noise = noise.mean(0, keepdim=True) if noise.shape[0] > 1 else noise
        noise = self._crop_pad(noise, length=self.segment)
        snr_db = random.uniform(-5, 20)
        noisy = self._mix(clean, noise, snr_db)

        if random.random() < 0.5 and len(self.rir) > 0:
            r_path = random.choice(self.rir)
            rir, _ = torchaudio.load(r_path)
            rir = rir.mean(0)[:8000]
            if rir.numel() > 0:
                rir = rir / (rir.abs().max() + 1e-8)
                noisy = torch.nn.functional.conv1d(noisy.unsqueeze(0), rir.view(1, 1, -1), padding=rir.shape[0] // 2).squeeze(0)[:, :self.segment]
                clean = torch.nn.functional.conv1d(clean.unsqueeze(0), rir.view(1, 1, -1), padding=rir.shape[0] // 2).squeeze(0)[:, :self.segment]

        peak = max(float(noisy.abs().max()), float(clean.abs().max()))
        if peak > 0.95:
            scale = 0.95 / peak
            noisy = noisy * scale
            clean = clean * scale
        return {"noisy": noisy, "clean": clean, "audiogram": ag, "snr": torch.tensor(snr_db, dtype=torch.float32), "file": self.clean[idx]}

    def _crop_pad(self, wav, length=None):
        L = length or self.segment
        if wav.shape[-1] > L:
            s = torch.randint(0, wav.shape[-1] - L, (1,)).item()
            return wav[:, s:s+L]
        elif wav.shape[-1] < L:
            return torch.nn.functional.pad(wav, (0, L - wav.shape[-1]))
        return wav

    def _mix(self, clean, noise, snr_db):
        ce = (clean ** 2).mean()
        ne = (noise ** 2).mean() + 1e-8
        scale = (ce / (ne * 10 ** (snr_db / 10))) ** 0.5
        return clean + scale * noise


def _no_dot(files):
    return [f for f in files if not os.path.basename(f).startswith("._") and "/._" not in f]


def build_filelists(data_dir, pattern_clean="voicebank/clean_trainset_28spk_wav/*.wav",
                   pattern_noise="musan/**/*.wav", pattern_rir="rir/**/*.wav"):
    """Maps to data/processed tree (verified counts Oct 15):
       clean:  11071  voicebank/clean_trainset_28spk_wav/*.wav   (11572 - 500 dot ghosts = 11071 real)
       noise:    930  musan/noise/**.wav
       rir:    60000  rir/**.wav  (largeroom/mediumroom/smallroom, dotfiles excluded)
       NOIZEUS (clean/noizeus) is eval-only, not training clean."""
    import glob
    # Clean = VoiceBank clean_trainset_28spk_wav only (NOIZEUS excluded from training)
    clean_candidates = sorted(glob.glob(os.path.join(data_dir, pattern_clean), recursive=True))
    if not clean_candidates:
        alt = sorted(glob.glob(os.path.join(data_dir, "voicebank/clean_*.wav"), recursive=True))
        alt += sorted(glob.glob(os.path.join(data_dir, "**/clean_trainset*/**/*.wav"), recursive=True))
        clean_candidates = alt
    clean = _no_dot(clean_candidates)

    # Noise = MUSAN only (not RIR, not VoiceBank noisy — those have speech leakage)
    noise_candidates = sorted(glob.glob(os.path.join(data_dir, pattern_noise), recursive=True))
    noise = _no_dot(noise_candidates)

    # RIR = openSLR26 simulated (rir/ contains largeroom/mediumroom/smallroom)
    rir_candidates = sorted(glob.glob(os.path.join(data_dir, pattern_rir), recursive=True))
    rir = _no_dot(rir_candidates)
    return clean, noise, rir
