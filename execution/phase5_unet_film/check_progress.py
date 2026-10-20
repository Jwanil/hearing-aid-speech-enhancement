# Helper script to monitor Phase A training progress
import os, sys, time, torch

ckpt_dir = "results/phase5_unet_film/checkpoints"
best_path = os.path.join(ckpt_dir, "best_composite.pt")
last_path = os.path.join(ckpt_dir, "last.pt")

print("=" * 60)
print("  Phase A Training Progress Monitor (Local RTX 4090)")
print("=" * 60)

if not os.path.exists(last_path):
    print("No checkpoints found yet. Training is preparing or starting.")
    sys.exit(0)

try:
    last_ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
    best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False) if os.path.exists(best_path) else last_ckpt

    cur_epoch = last_ckpt.get("epoch", 0)
    total_epochs = 25
    last_val = last_ckpt.get("val_loss", 0.0)
    best_val = best_ckpt.get("val_loss", 0.0)
    best_epoch = best_ckpt.get("epoch", 0)

    mtime = os.path.getmtime(last_path)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))

    print(f"Current Epoch:      {cur_epoch} / {total_epochs} ({(cur_epoch/total_epochs)*100:.1f}%)")
    print(f"Last Val Loss:      {last_val:.4f}")
    print(f"Best Val Loss:      {best_val:.4f} (at Epoch {best_epoch})")
    print(f"Last Checkpoint:    {time_str}")

    if cur_epoch >= total_epochs:
        print("\n[SUCCESS] Training is COMPLETE! All 25 epochs finished.")
    else:
        rem_epochs = total_epochs - cur_epoch
        est_min = rem_epochs * 2.5
        print(f"Estimated Time Left: ~{est_min:.0f} minutes (~{est_min/60:.1f} hours)")
except Exception as e:
    print(f"Error reading checkpoint: {e}")

print("=" * 60)
