"""
Verify Setup Script
-------------------
Run this first to confirm all dependencies are installed correctly.
Usage: python execution/00_verify_setup.py
"""

import sys

checks = []

def assert_version():
    """Check Python version is 3.9+"""
    major, minor = sys.version_info[:2]
    assert (major, minor) >= (3, 9), f"Need Python 3.9+, got {major}.{minor}"

def check(name, fn):
    try:
        fn()
        checks.append((name, True, ""))
    except Exception as e:
        checks.append((name, False, str(e)))

# --- Python version ---
check("Python 3.9+", assert_version)

# --- Core scientific stack ---
check("torch", lambda: __import__("torch"))
check("torchaudio", lambda: __import__("torchaudio"))
check("numpy", lambda: __import__("numpy"))
check("scipy", lambda: __import__("scipy"))
check("matplotlib", lambda: __import__("matplotlib"))

# --- Audio I/O ---
check("librosa", lambda: __import__("librosa"))
check("soundfile", lambda: __import__("soundfile"))

# --- Wavelet (Phase 3 — Classical Baselines) ---
check("pywavelets (PyWavelets)", lambda: __import__("pywt"))

# --- Speech enhancement metrics ---
check("pystoi", lambda: __import__("pystoi"))
check("torchmetrics (replaces pesq)", lambda: __import__("torchmetrics"))

# --- Speech enhancement framework ---
check("speechbrain", lambda: __import__("speechbrain"))

# --- Hearing-aid specific ---
check("pyclarity (clarity.evaluator)", lambda: __import__("clarity.evaluator.msbg.msbg", fromlist=["Ear"]))

# --- Jupyter (optional — not required for scripts) ---
check("jupyter / notebook (optional)", lambda: __import__("notebook"))

# --- Print results ---
print("\n" + "="*52)
print("  ENVIRONMENT VERIFICATION")
print("="*52)

OPTIONAL = {"jupyter / notebook (optional)"}

all_pass = True
for name, passed, err in checks:
    is_optional = name in OPTIONAL
    if passed:
        status = "✅"
    elif is_optional:
        status = "⚠️ "
    else:
        status = "❌"
    print(f"  {status}  {name}")
    if not passed:
        print(f"       └─ {'(optional) ' if is_optional else ''}ERROR: {err}")
        if not is_optional:
            all_pass = False

print("="*52)
if all_pass:
    print("  ✅  All checks passed! Environment is ready.")
else:
    print("  ⚠️   Some checks failed. Run:")
    print("  ~/Library/Python/3.9/bin/pip3 install torch torchaudio")
    print("       speechbrain pyclarity pystoi torchmetrics")
    print("       librosa soundfile PyWavelets scipy matplotlib")
print("="*52 + "\n")
