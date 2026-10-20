"""Side-effect-free smoke checks for the corrected Phase 5A eval protocol.

This test creates a tiny temporary VoiceBank-shaped directory, verifies sorted
clean/noisy pairing and orphan rejection, and then removes the temporary tree.
It does not need the real SSD dataset or a CUDA device.
"""

import importlib.util
import tempfile
import wave
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_data_module():
    path = HERE / "03c_data_pipeline_v2.py"
    spec = importlib.util.spec_from_file_location("phase5_data_protocol", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * 160)


def main():
    data = load_data_module()
    with tempfile.TemporaryDirectory(prefix="phase5a_protocol_") as temp:
        root = Path(temp) / "processed" / "voicebank"
        clean_dir = root / "clean_testset_wav"
        noisy_dir = root / "noisy_testset_wav"
        for name in ("p257_002.wav", "p232_001.wav"):
            write_wav(clean_dir / name)
            write_wav(noisy_dir / name)

        pairs = data.build_voicebank_test_pairs(str(Path(temp) / "processed"))
        assert [Path(clean).name for clean, _ in pairs] == ["p232_001.wav", "p257_002.wav"]

        write_wav(noisy_dir / "orphan.wav")
        try:
            data.build_voicebank_test_pairs(str(Path(temp) / "processed"))
        except ValueError as exc:
            assert "missing_clean=1" in str(exc)
        else:
            raise AssertionError("Orphan noisy test file was not rejected")

    print("PHASE5A EVALUATION PROTOCOL PASS")


if __name__ == "__main__":
    main()
