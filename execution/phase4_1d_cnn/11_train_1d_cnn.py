import os
import importlib.util
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from asteroid.losses import pairwise_neg_sisdr
from datasets import load_dataset
from tqdm.auto import tqdm

# Import model definition via importlib (avoids Python's digit-in-module-name restriction)
_spec = importlib.util.spec_from_file_location(
    "model_1d_cnn",
    os.path.join(os.path.dirname(__file__), "10_model_1d_cnn.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
get_1d_cnn_model = _mod.get_1d_cnn_model

# ── Configuration ─────────────────────────────────────────────────────────────
# batch=8 fits T4 (16GB) and cuts per-epoch time ~2x vs batch=4
# 30 epochs × ~15 min/epoch ≈ 7.5 hrs — comfortably within one Colab session
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 3e-4
SEGMENT_LENGTH = 64000  # 4 seconds at 16kHz
CKPT_DIR = "results/checkpoints"

# ── Dataset Definition ───────────────────────────────────────────────────────
class VoiceBankDemand16k(Dataset):
    """PyTorch Dataset wrapper for HuggingFace VoiceBank-DEMAND."""
    def __init__(self, hf_dataset, segment_length=64000):
        self.dataset = hf_dataset
        self.segment_length = segment_length
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        clean = torch.tensor(self.dataset[idx]['clean']['array'], dtype=torch.float32)
        noisy = torch.tensor(self.dataset[idx]['noisy']['array'], dtype=torch.float32)
        
        # Pad or trim to segment_length
        if len(clean) > self.segment_length:
            # Random crop for training robustness
            start = torch.randint(0, len(clean) - self.segment_length, (1,)).item()
            clean = clean[start:start+self.segment_length]
            noisy = noisy[start:start+self.segment_length]
        elif len(clean) < self.segment_length:
            # Pad with zeros
            pad_len = self.segment_length - len(clean)
            clean = torch.nn.functional.pad(clean, (0, pad_len))
            noisy = torch.nn.functional.pad(noisy, (0, pad_len))
            
        # Add channel dimension: (1, T)
        return noisy.unsqueeze(0), clean.unsqueeze(0)

# ── Main Training Loop ───────────────────────────────────────────────────────
def train():
    os.makedirs(CKPT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Loading JacobLinCool/VoiceBank-DEMAND-16k from Hugging Face...")
    ds = load_dataset('JacobLinCool/VoiceBank-DEMAND-16k')
    
    train_ds = VoiceBankDemand16k(ds['train'], SEGMENT_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # 2. Build Model
    model = get_1d_cnn_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Train
    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for noisy, clean in pbar:
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            est_clean = model(noisy)
            
            # Calculate Negative SI-SDR loss
            loss = pairwise_neg_sisdr(est_clean, clean).mean()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (standard for TCNs)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.2f}"})
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} Average Loss (Negative SI-SDR): {avg_loss:.2f}")
        
        # Save every epoch — protects against Colab session resets
        ckpt_path = os.path.join(CKPT_DIR, f"1d_cnn_epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    train()
