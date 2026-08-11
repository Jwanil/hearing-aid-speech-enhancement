# Setup Guide for Namya Shah — Hearing Aid ML Project

> Hi Namya! This guide gets you from zero to fully set up and coding alongside Jwanil.
> Read this **completely** before doing anything else. It takes about 30–45 minutes.

---

## What You're Joining

We're building a machine learning model that personalizes noise reduction in hearing aids based on a user's individual hearing-loss profile (audiogram). The project uses Python + PyTorch + deep learning.

**Before diving in, read:** `docs/project_overview.md` for the full technical context, and `docs/everything_from_scratch.md` for a from-scratch explanation of every concept.

---

## Step 1: Install VS Code

If you don't have it already:
- Download from: https://code.visualstudio.com/
- Install normally, launch it

---

## Step 2: Install Extensions

You need two extensions:

**Live Share** — real-time collaborative coding with Jwanil:
1. Press `Cmd+Shift+X` (Mac) or `Ctrl+Shift+X` (Windows/Linux)
2. Search: **"Live Share"** by Microsoft → Install
3. Restart VS Code
4. Sign in with your **GitHub account** when prompted (bottom left of VS Code)

**Five Server** — local development server with live reload:
1. Same Extensions panel
2. Search: **"Five Server (Live Server)"** → Install
3. You'll use this later to preview the `presentation.html` demo

**To join a Live Share session when Jwanil shares a link:**
- Click the link → opens VS Code and connects you instantly
- You'll see Jwanil's files and cursor in your editor in real-time

---

## Step 3: Install Git

Check if you have it:
```bash
git --version
```

If not:
- **Mac:** `xcode-select --install`  
- **Windows:** https://git-scm.com/download/win
- **Linux:** `sudo apt install git`

Configure your identity:
```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

---

## Step 4: Clone the Project

*(Jwanil will give you the GitHub link — replace below)*

```bash
git clone https://github.com/Jwanil/hearing-aid-speech-enhancement.git
cd hearing-aid-speech-enhancement
```

---

## Step 5: Install Python & Dependencies

**Check Python version (need 3.9+):**
```bash
python3 --version
```

**Create a virtual environment (important — keeps this project isolated):**
```bash
python3 -m venv venv
source venv/bin/activate     # Mac/Linux
# OR
venv\Scripts\activate        # Windows
```

**Install all dependencies:**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install speechbrain asteroid-filterbanks pyclarity pystoi
pip install librosa soundfile matplotlib numpy scipy pandas
pip install jupyter ipykernel tensorboard
```

> ⚠️ If you have an NVIDIA GPU, install the CUDA version of torch instead — ask Jwanil or look up the correct command at pytorch.org/get-started.

**Verify everything works:**
```bash
python execution/00_verify_setup.py
```

You should see all green checkmarks. If something fails, check the error message and Slack/WhatsApp Jwanil.

---

## Step 6: Install Antigravity (Your AI Pair Programmer)

Antigravity is the AI tool we use to help write and debug code. **You each have your own instance** — it runs on your machine, not shared.

1. Go to: **https://antigravity.google**
2. Download and install for your OS
3. Sign in with your Google account
4. Open Antigravity
5. Set the workspace: **Projects → Add Project → select the `hearing-aid-speech-enhancement` folder you cloned**

**First session — paste this into Antigravity chat:**
```
I'm joining an existing ML project called "Hearing Aid Speech Enhancement."
Please read these files to get context:
1. docs/project_overview.md
2. context.md  
3. shared_context.md

After reading, tell me: what phase are we in, what has been done, and what should I work on next?
```

The agent will orient itself and tell you exactly where to start.

---

## Step 7: Open the Project in VS Code

```bash
code "/path/to/hearing-aid-speech-enhancement"
```

Or: File → Open Folder → select the project folder.

---

## Step 8: Update the Team Table

Open `shared_context.md` and fill in your row in the Team table at the top:

```markdown
| [Your Name] | Co-developer | [Your OS] | Installed | @[your-github-handle] |
```

Commit and push:
```bash
git add shared_context.md
git commit -m "chore: add [your name] to team table"
git push
```

---

## Day-to-Day Workflow

### When Jwanil is online (Live Share session):
1. Jwanil shares a Live Share link in your group chat
2. Click the link → you're in the same VS Code session
3. Code together in real-time
4. Either of you can chat with your own Antigravity while the other works
5. When done, whoever made changes commits and pushes

### When working independently (async):
1. Pull latest: `git pull`
2. Open your Antigravity workspace
3. Read `context.md` and `shared_context.md` to see what's been done
4. Work on your assigned task (check `shared_context.md` → Partner Responsibilities table)
5. After every significant action, update both logs
6. Commit and push: `git add . && git commit -m "describe what you did" && git push`
7. Notify Jwanil in the group chat

---

## Key Files to Know

| File | What it is |
|------|-----------|
| `docs/project_overview.md` | Full project background — read this first |
| `docs/everything_from_scratch.md` | Theory explainer — all concepts from basics |
| `docs/presentation.html` | Faculty pitch deck — open in browser |
| `context.md` | What's been done, current tasks — always check this |
| `shared_context.md` | What you and Jwanil have each done |
| `AGENTS.md` | Instructions for the AI agent |
| `directives/` | Step-by-step guides for each project phase |
| `execution/` | Python scripts that do the actual work |

---

## If You Get Stuck

1. **Read the relevant directive** in `directives/` for what you're working on
2. **Ask your Antigravity agent** — give it the context files listed in Step 6
3. **Check `context.md`** — maybe the problem is already solved
4. **Message Jwanil**

---

## Quick Reference Commands

```bash
# Activate virtual environment (do this every time you open a terminal)
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Pull latest changes
git pull

# Commit and push your work
git add .
git commit -m "your message"
git push

# Run a script
python execution/SCRIPT_NAME.py

# Start Jupyter notebook
jupyter notebook
```

---

Good luck! The project is well-scaffolded — the hardest part is already done. 🎧
