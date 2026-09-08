# Google Colab + Antigravity IDE Setup Guide

> **Who this is for:** Both Jwanil and Namya. Read this once, set it up once.
> After setup, the agent can write and run training code on Colab's GPU directly from the IDE.

---

## What This Gives You

Instead of manually writing notebook cells and switching between the IDE and Colab, the
`colab-mcp` bridge lets the AI agent:
- Execute Python code on Colab's T4/A100 GPU directly from the chat
- Read training logs and loss curves back into the IDE
- Run long training loops without you watching the browser

---

## Architecture

```
Antigravity IDE (your local machine)
        │
        │  colab-mcp MCP bridge (stdio)
        ▼
Colab-MCP Chrome Extension (browser)
        │
        ▼
Google Colab Notebook (GPU runtime)
  ├── Git clone of this repo (code)
  └── Google Drive mount (data — 930 WAVs)
```

---

## One-Time Setup: 3 Parts

---

### Part A — Agent Side (already done if you cloned the repo)

The MCP config is committed at `.agents/mcp_config.json`. Antigravity picks it up automatically when you open this workspace.

**Check it's working:** In a new Antigravity session, you should see `colab-mcp` listed in available tools. If not, restart the IDE.

**Requires `uv` to be installed on your machine:**
```bash
pip install uv
# or
brew install uv
```

---

### Part B — Browser Side (each partner does this once)

#### 1. Install the Colab-MCP Chrome Extension

Go to: **https://chromewebstore.google.com/detail/colab-mcp/aahmgnbckmpnhckaplnddkamoldcgnkj**

Install it. No configuration needed — it auto-connects when you open Colab.

> **Note:** Works in Chrome and Chrome-based browsers (Edge, Brave). Not supported in Firefox or Safari.

#### 2. Open a Colab notebook and connect

1. Go to **https://colab.research.google.com**
2. Create a new notebook (or open an existing one)
3. Click **Runtime → Change runtime type → T4 GPU** (free) or A100 (Colab Pro)
4. Click **Connect** (top right)
5. The extension icon in your browser toolbar should show a green dot — this means the bridge is live

---

### Part C — Data Setup (each partner does this once on their Google account)

The 930 processed NOIZEUS WAVs need to be on Google Drive so Colab can access them.

#### Jwanil — upload from SSD
```bash
# Run this once on your Mac to upload the processed folder to Drive
# Install rclone first: brew install rclone
# Then configure: rclone config (choose Google Drive, follow prompts)

rclone copy "/Volumes/SANDISK/Minor Project/Data/processed" \
  "gdrive:hearing-aid-data/processed" \
  --progress
```

Or just drag-and-drop `data/processed/` into Google Drive manually (it's ~200MB for NOIZEUS).

#### Namya — upload from local clone
Same as above, but source is your local `dataset/processed/` folder:
```bash
rclone copy "dataset/processed" "gdrive:hearing-aid-data/processed" --progress
```

**Target Drive path:** `MyDrive/hearing-aid-data/processed/`

Expected structure after upload:
```
MyDrive/
└── hearing-aid-data/
    └── processed/
        ├── clean/
        │   └── noizeus/clean/*.wav    (30 files)
        └── noisy/
            └── noizeus/<noise_type>/<snr>/*.wav    (930 files)
```

---

## How to Use It (Every Session)

1. Open Colab in Chrome with the extension installed
2. Make sure your notebook has a GPU runtime connected
3. Open Antigravity IDE in this workspace
4. Tell the agent: **"Open a Colab session and run the Phase 4 training"**

The agent will:
- Send a `!git clone` command to Colab
- Mount your Drive
- Install dependencies
- Run the training script
- Report back loss values, checkpoints saved, etc.

---

## First Colab Cell Template

The agent will generate this automatically, but here it is for reference:

```python
# ── Cell 1: Setup ──────────────────────────────────────────
import os

# Clone repo (always gets latest code from GitHub)
if not os.path.exists("hearing-aid-speech-enhancement"):
    !git clone https://github.com/Jwanil/hearing-aid-speech-enhancement.git
%cd hearing-aid-speech-enhancement
!git pull   # ensure latest

# Mount Drive (your 930 WAVs live here)
from google.colab import drive
drive.mount('/content/drive')

# Symlink data so scripts find it at the expected path
if not os.path.exists("data"):
    os.symlink("/content/drive/MyDrive/hearing-aid-data", "data")

# Install dependencies
!pip install -q pywt soundfile torchaudio pyclarity pystoi pesq mamba-ssm

print("Setup complete ✅")
print("Data path:", os.listdir("data/processed/noisy/noizeus"))
```

---

## Saving Checkpoints

Checkpoints are saved to Drive so they survive Colab session resets:

```python
# In training script — always save to Drive, not /content
CKPT_DIR = "/content/drive/MyDrive/hearing-aid-data/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

torch.save(model.state_dict(), f"{CKPT_DIR}/phase4_cnn_epoch{epoch}.pt")
```

After training, download the checkpoint locally:
```bash
# On your Mac (using rclone):
rclone copy "gdrive:hearing-aid-data/checkpoints/" results/checkpoints/
```

---

## Committing Results from Colab

After training, push results back to GitHub from inside Colab:

```python
# Configure git identity in Colab (do once per session)
!git config user.email "your@email.com"
!git config user.name "Jwanil Modi"

# Add a GitHub token for push auth (use a fine-grained token with repo scope)
import os
GITHUB_TOKEN = "ghp_..."  # paste your token — don't commit this!
!git remote set-url origin https://{GITHUB_TOKEN}@github.com/Jwanil/hearing-aid-speech-enhancement.git

# Commit and push results
!git add results/
!git commit -m "phase4: training results from Colab [auto]"
!git push
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Extension icon is grey (not green) | Refresh the Colab tab; make sure the notebook runtime is connected |
| Agent can't find colab-mcp tools | Restart Antigravity IDE; check `uv` is installed |
| Drive not mounting | Re-run the `drive.mount()` cell; authorize in the popup |
| `data/` symlink already exists | `os.remove("data")` then re-create |
| Checkpoint lost after session | Always save to Drive path, not `/content/` |
| Namya can't push to repo | Add Namya as a collaborator: GitHub → Settings → Collaborators |

---

## Namya-Specific Notes

- Your data is in `dataset/processed/` locally — use that path when uploading to Drive
- Your IDE is Claude Code — the `.agents/mcp_config.json` in this repo will configure colab-mcp for you automatically when you open the workspace
- Your context file is `context.md` — keep it local, never commit it
- The Colab notebook is **shared** — you can open the same notebook Jwanil is using in your browser and both work on it simultaneously (Google Docs-style collaboration)

