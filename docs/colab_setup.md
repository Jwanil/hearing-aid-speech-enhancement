# Google Colab + Antigravity IDE Setup Guide

> **Who this is for:** Both Jwanil and Namya. Read once, set up once.
> After setup, the AI agent can write and execute training code on Colab's GPU directly from the IDE.

---

## How It Works (Architecture)

```
Antigravity / Claude Code IDE
        │
        │  colab-mcp MCP server (runs via uvx on your Mac)
        │  — exposes tools: open_colab_browser_connection, execute_python, etc.
        ▼
WebSocket server (localhost, random port, secure token)
        │
        │  Colab notebook connects back via WebSocket
        ▼
Google Colab (browser, GPU runtime)
  ├── Code sent here by the agent, executed on T4/A100
  └── Output returned back to the agent in real-time
```

**No Chrome extension required.** The bridge works entirely over a local WebSocket.
When the agent calls `open_colab_browser_connection`, it:
1. Starts a local WebSocket server with a one-time secure token
2. Opens a Colab URL in your browser with the token embedded
3. Colab connects back to your machine
4. Live bridge established — agent can now execute cells on the GPU

---

## One-Time Setup

### Prerequisites (both partners)

**1. Install `uv`**
```bash
pip install uv
# macOS alternative:
brew install uv
```
`uv` manages Python 3.13+ automatically — colab-mcp requires Python 3.13+.

**2. MCP config** (already in the repo — nothing to do)

The file `.agents/mcp_config.json` is committed in this repo:
```json
{
  "mcpServers": {
    "colab-mcp": {
      "command": "uvx",
      "args": ["git+https://github.com/googlecolab/colab-mcp"],
      "timeout": 30000
    }
  }
}
```
Antigravity (Jwanil) and Claude Code (Namya) both auto-load this when the workspace is opened.

**3. Restart your IDE** after pulling the repo to activate the MCP.

---

## Data Setup — Google Drive (each partner, once)

The 930 processed NOIZEUS WAVs need to be on your Google Drive so Colab can mount them.

### Jwanil — upload from SSD
```bash
# Option A: drag-and-drop in Google Drive browser
# Upload: /Volumes/SANDISK/Minor Project/Data/processed/
# Destination: MyDrive/hearing-aid-data/processed/

# Option B: rclone (faster for large folders)
brew install rclone
rclone config   # choose Google Drive, follow OAuth prompts, name it "gdrive"
rclone copy "/Volumes/SANDISK/Minor Project/Data/processed" \
  "gdrive:hearing-aid-data/processed" --progress
```

### Namya — upload from local clone
```bash
rclone copy "dataset/processed" "gdrive:hearing-aid-data/processed" --progress
```

**Expected Drive structure after upload:**
```
MyDrive/
└── hearing-aid-data/
    └── processed/
        ├── clean/noizeus/clean/*.wav       (30 files)
        └── noisy/noizeus/<noise>/<snr>/*.wav   (930 files)
```

---

## Starting a Colab Session (Every Time)

1. Make sure your IDE is open in this workspace with colab-mcp active
2. Tell the agent: **"Open a Colab connection"**
3. The agent calls `open_colab_browser_connection` — a Colab URL opens in your browser
4. In Colab: **Runtime → Change runtime type → T4 GPU → Save**
5. Click **Connect** (top right of Colab)
6. The agent confirms the connection and begins executing cells

---

## First Session Setup Cell (agent sends this automatically)

```python
# ── Setup: Clone repo + mount Drive + install deps ──────────────
import os

# Always get latest code
if not os.path.exists("hearing-aid-speech-enhancement"):
    !git clone https://github.com/Jwanil/hearing-aid-speech-enhancement.git
%cd hearing-aid-speech-enhancement
!git pull

# Mount Google Drive (your 930 WAVs)
from google.colab import drive
drive.mount('/content/drive')

# Symlink Drive data to expected path
if os.path.exists("data"):
    os.remove("data") if os.path.islink("data") else None
os.symlink("/content/drive/MyDrive/hearing-aid-data", "data")

# Verify data is accessible
print("Noise types:", sorted(os.listdir("data/processed/noisy/noizeus")))

# Install project dependencies
!pip install -q pywt soundfile torchaudio pyclarity pystoi pesq

print("Setup complete ✅")
```

---

## Saving Checkpoints (Never Lose Training Progress)

Colab sessions reset after ~12h or on disconnect. Always save to Drive:

```python
CKPT_DIR = "/content/drive/MyDrive/hearing-aid-data/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

# In your training loop:
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss,
}, f"{CKPT_DIR}/phase4_cnn_epoch{epoch:03d}.pt")
```

Download checkpoints locally when training is done:
```bash
# Mac (rclone):
rclone copy "gdrive:hearing-aid-data/checkpoints/" results/checkpoints/
```

---

## Committing Results Back to GitHub (from Colab)

```python
# Set git identity in Colab
!git config user.email "23bit013@ldrp.ac.in"
!git config user.name "Jwanil Modi"   # or Namya Shah

# Use a GitHub fine-grained personal access token (repo scope)
# Generate at: github.com → Settings → Developer Settings → Tokens
import os
token = "ghp_..."  # paste your token — DO NOT commit this to the repo
!git remote set-url origin https://{token}@github.com/Jwanil/hearing-aid-speech-enhancement.git

# Push results
!git add results/ shared_context.md
!git commit -m "phase4: training results from Colab [Jwanil]"
!git push
```

---

## Namya-Specific Notes

- Your data path locally: `dataset/processed/` (not `data/`) — use this when uploading to Drive
- Your IDE: Claude Code — `.agents/mcp_config.json` is auto-loaded from the repo
- Your context file: `context.md` — local only, never commit it
- Both partners can have the **same Colab notebook open simultaneously** — Google Colab supports real-time collaboration like Google Docs
- Ask Jwanil to share the Colab notebook link the first time

---

## Troubleshooting

| Problem | Fix |
|---|---|
| colab-mcp not in agent tools | Restart IDE; ensure `uv` is installed (`uv --version`) |
| Colab URL doesn't open | Call `open_colab_browser_connection` again; check browser popup blocker |
| Drive not mounting | Re-run `drive.mount('/content/drive')` and authorize in the popup |
| Data not found | Check symlink: `ls -la data/` should point to Drive |
| Checkpoint lost | You saved to `/content/` not Drive — always use `CKPT_DIR` |
| Push rejected from Colab | Token expired or missing repo scope — generate a new one |
| "requires Python >=3.13" error | `uvx` handles this automatically — no action needed |

---

## Quick Reference

| Task | Command / Action |
|---|---|
| Start Colab session | Tell agent: "Open a Colab connection" |
| Check GPU available | `!nvidia-smi` in Colab |
| Get latest code | `!git pull` in Colab |
| Save checkpoint | `torch.save(...)` to `CKPT_DIR` |
| Download checkpoint | `rclone copy "gdrive:hearing-aid-data/checkpoints/" results/checkpoints/` |
| Check Drive data | `os.listdir("data/processed/noisy/noizeus")` |
