import os
import glob
import multiprocessing
import torchaudio
import torch
import soundfile as sf
import argparse
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")

TARGET_SR = 16000     # target sampling rate: 16,000 Hz (16 kHz)
TARGET_LEN = 64000    # target length: 4 seconds at 16kHz = 64,000 samples

def standardize(waveform, src_sr):
    """
    Bring any .wav file to the project standard:
    - Mono (1 channel)
    - 16,000 Hz sampling rate
    - Amplitude normalized to [-1, +1]
    - Length = TARGET_LEN samples (pad with silence or trim)
    """
    # 1. Mono conversion
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 2. Resample to 16kHz if needed
    if src_sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(src_sr, TARGET_SR)
        waveform = resampler(waveform)

    # 3. Normalize amplitude to [-1, +1]
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak

    # 4. Fix length
    length = waveform.shape[1]
    if length > TARGET_LEN:
        waveform = waveform[:, :TARGET_LEN]
    elif length < TARGET_LEN:
        pad = TARGET_LEN - length
        waveform = torch.nn.functional.pad(waveform, (0, pad))

    return waveform

def process_file(file_path):
    # Compute output path by swapping raw with processed
    rel_path = os.path.relpath(file_path, RAW_DIR)
    out_path = os.path.join(PROCESSED_DIR, rel_path)
    
    # Skip if already exists
    if os.path.exists(out_path):
        return True
        
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    try:
        # Load audio (returns shape [channels, frames])
        waveform, sr = torchaudio.load(file_path)
        
        # Apply standardisation
        processed = standardize(waveform, sr)
        
        # Save as 16-bit PCM wav
        sf.write(out_path, processed.squeeze(0).numpy(), TARGET_SR, subtype='PCM_16')
        return True
    except Exception as e:
        print(f"\nError processing {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Standardize all raw audio files")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count() - 1, help="Number of parallel workers")
    args = parser.parse_args()
    
    print("="*60)
    print(" PHASE 2: AUDIO STANDARDIZATION (16kHz, Mono, 4s, Normalized)")
    print("="*60)
    
    print("Scanning for .wav files in data/raw/ ...")
    
    all_files = []
    for root, _, files in os.walk(RAW_DIR):
        for f in files:
            if f.endswith(".wav") and not f.startswith("._"):
                all_files.append(os.path.join(root, f))
                
    print(f"Found {len(all_files)} files. Standardizing to data/processed/ ...")
    print(f"Using {args.workers} workers for parallel processing.")
    
    success_count = 0
    with multiprocessing.Pool(args.workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_file, all_files), total=len(all_files), desc="Processing"))
        
    success_count = sum(1 for r in results if r)
    
    print("="*60)
    print(f"✅ Processed {success_count} / {len(all_files)} files successfully.")
    print("Output directory: data/processed/")
    print("="*60)

if __name__ == "__main__":
    main()
