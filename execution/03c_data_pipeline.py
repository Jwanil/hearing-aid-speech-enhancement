import os
import csv
import json
import random
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
META_DIR = os.path.join(ROOT, "data", "metadata")
AUDIOGRAMS_JSON = os.path.join(ROOT, "results", "data", "audiograms.json")

TARGET_SR = 16000
TARGET_LEN = 64000

class HearingAidDataset(Dataset):
    def __init__(self, manifest_file, split="train"):
        self.manifest_path = os.path.join(META_DIR, manifest_file)
        self.split = split
        self.items = []
        
        # Load the CSV
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
            
        with open(self.manifest_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.items.append(row)
                
        # Load the audiograms
        if os.path.exists(AUDIOGRAMS_JSON):
            with open(AUDIOGRAMS_JSON, "r") as f:
                ag_data = json.load(f)
                self.audiograms = {}
                for p in ag_data.get("test_profiles", []):
                    self.audiograms[p["name"]] = p["levels"]
                for i, p in enumerate(ag_data.get("random_audiograms", [])):
                    self.audiograms[f"random_{i}"] = p["levels"]
        else:
            print(f"Warning: {AUDIOGRAMS_JSON} not found. Using flat 0dB audiograms.")
            self.audiograms = {"default": [0, 0, 0, 0, 0, 0]}

        self.audiogram_keys = list(self.audiograms.keys())

    def __len__(self):
        return len(self.items)

    def load_audio(self, rel_path):
        """Load audio from data/processed/ (must already be 16kHz, mono, 64k length)"""
        full_path = os.path.join(PROCESSED_DIR, rel_path)
        if not os.path.exists(full_path):
            # Fallback to zeros if standardisation hasn't finished for this file yet
            return torch.zeros(1, TARGET_LEN)
            
        try:
            waveform, sr = torchaudio.load(full_path)
            # Ensure it strictly matches 1x64000
            if waveform.shape[1] > TARGET_LEN:
                waveform = waveform[:, :TARGET_LEN]
            elif waveform.shape[1] < TARGET_LEN:
                pad = TARGET_LEN - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad))
            return waveform
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            return torch.zeros(1, TARGET_LEN)

    def __getitem__(self, idx):
        item = self.items[idx]
        
        noisy = self.load_audio(item["filename"])
        clean = self.load_audio(item["clean_file"])
        
        # Assign a random audiogram for training to simulate diverse hearing loss profiles
        ag_name = random.choice(self.audiogram_keys)
        ag_thresholds = self.audiograms[ag_name]
        
        # Normalize audiogram from [0, 120] dB to [0, 1] range for the neural net
        ag_tensor = torch.tensor(ag_thresholds, dtype=torch.float32) / 120.0
        
        snr = item["snr_db"]
        if snr == "unknown":
            snr = 0.0
        else:
            try:
                snr = float(snr)
            except ValueError:
                snr = 0.0

        return {
            'noisy': noisy,
            'clean': clean,
            'audiogram': ag_tensor,
            'snr': snr,
            'filename': item["filename"]
        }


def test_dataloader():
    print("="*60)
    print(" TESTING DATALOADER")
    print("="*60)
    
    # Use val manifest for a quick test
    dataset = HearingAidDataset("val_manifest.csv", split="val")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"Dataset length: {len(dataset)} items")
    
    batch = next(iter(loader))
    
    print("Batch dict keys:", batch.keys())
    print("Noisy shape:", batch['noisy'].shape)
    print("Clean shape:", batch['clean'].shape)
    print("Audiogram shape:", batch['audiogram'].shape)
    
    assert batch['noisy'].shape == (4, 1, 64000), "Noisy shape mismatch!"
    assert batch['clean'].shape == (4, 1, 64000), "Clean shape mismatch!"
    assert batch['audiogram'].shape == (4, 6), "Audiogram shape mismatch!"
    
    # Plot spectrograms of the first item in the batch
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.specgram(batch['clean'][0, 0].numpy(), Fs=16000, NFFT=512, noverlap=256)
    plt.title(f"Clean Speech (File: {batch['filename'][0]})")
    plt.ylabel("Frequency (Hz)")
    
    plt.subplot(2, 1, 2)
    plt.specgram(batch['noisy'][0, 0].numpy(), Fs=16000, NFFT=512, noverlap=256)
    plt.title(f"Noisy Speech (SNR: {batch['snr'][0]} dB)")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    
    plt.tight_layout()
    plot_path = os.path.join(ROOT, "results", "plots", "02_dataloader_test.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    print(f"✅ Dataloader plot saved to {plot_path}")
    print("="*60)

if __name__ == "__main__":
    test_dataloader()
