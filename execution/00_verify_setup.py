"""
Verify Setup Script
-------------------
Run this first to confirm all dependencies are installed correctly.
Usage: python execution/00_verify_setup.py
"""

import sys

checks = []

def check(name, fn):
    try:
        fn()
        checks.append((name, True, ""))
    except Exception as e:
        checks.append((name, False, str(e)))

# Core
check("Python 3.9+", lambda: assert_version())
check("torch", lambda: __import__("torch"))
check("torchaudio", lambda: __import__("torchaudio"))
check("numpy", lambda: __import__("numpy"))
check("scipy", lambda: __import__("scipy"))
check("matplotlib", lambda: __import__("matplotlib"))

# Audio
check("librosa", lambda: __import__("librosa"))
check("soundfile", lambda: __import__("soundfile"))

# Speech enhancement
check("speechbrain", lambda: __import__("speechbrain"))

# Hearing-aid specific
check("pyclarity (clarity.evaluator)", lambda: __import__("clarity.evaluator.msbg.msbg", fromlist=["Ear"]))
check("pystoi", lambda: __import__("pystoi"))

# Jupyter
check("jupyter", lambda: __import__("notebook"))

def assert_version():
    major, minor = sys.version_info[:2]
    assert (major, minor) >= (3, 9), f"Need Python 3.9+, got {major}.{minor}"

# Re-run assert_version properly
try:
    assert_version()
    checks[0] = ("Python 3.9+", True, "")
except AssertionError as e:
    checks[0] = ("Python 3.9+", False, str(e))

# Print results
print("\n" + "="*50)
print("  ENVIRONMENT VERIFICATION")
print("="*50)

all_pass = True
for name, passed, err in checks:
    status = "✅" if passed else "❌"
    print(f"  {status}  {name}")
    if not passed:
        print(f"       ERROR: {err}")
        all_pass = False

print("="*50)
if all_pass:
    print("  All checks passed! Environment is ready.")
else:
    print("  Some checks failed. Install missing packages:")
    print("  pip install torch torchaudio speechbrain pyclarity pystoi librosa soundfile")
print("="*50 + "\n")
