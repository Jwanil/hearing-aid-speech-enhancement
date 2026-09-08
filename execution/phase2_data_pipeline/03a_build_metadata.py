import os
import csv
import glob
import torchaudio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
META_DIR = os.path.join(DATA_DIR, "metadata")

os.makedirs(META_DIR, exist_ok=True)

def get_audio_info(filepath):
    try:
        info = torchaudio.info(filepath)
        duration = info.num_frames / info.sample_rate
        return duration, info.sample_rate
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0.0, 0

def build_noizeus_metadata():
    print("Building NOIZEUS metadata...")
    noizeus_noisy_dir = os.path.join(RAW_DIR, "noisy", "noizeus")
    noizeus_clean_dir = os.path.join(RAW_DIR, "clean", "noizeus", "clean")
    
    rows = []
    # Noizeus structure: data/raw/noisy/noizeus/<noise_type>/<snr>dB/spXX_<noise>_sn<snr>.wav
    for noise_type in os.listdir(noizeus_noisy_dir):
        noise_dir = os.path.join(noizeus_noisy_dir, noise_type)
        if not os.path.isdir(noise_dir):
            continue
            
        for snr_folder in os.listdir(noise_dir):
            snr_dir = os.path.join(noise_dir, snr_folder)
            if not os.path.isdir(snr_dir):
                continue
                
            snr_val = snr_folder.replace("dB", "")
            
            for wav_file in os.listdir(snr_dir):
                if not wav_file.endswith(".wav") or wav_file.startswith("._"):
                    continue
                    
                noisy_path = os.path.join(snr_dir, wav_file)
                # Parse spXX
                speaker_id = wav_file.split("_")[0]
                clean_path = os.path.join(noizeus_clean_dir, f"{speaker_id}.wav")
                
                duration, sr = get_audio_info(noisy_path)
                
                # We save paths relative to data/raw/
                rel_noisy = os.path.relpath(noisy_path, RAW_DIR)
                rel_clean = os.path.relpath(clean_path, RAW_DIR)
                
                rows.append({
                    "filename": rel_noisy,
                    "clean_file": rel_clean,
                    "noise_type": noise_type,
                    "snr_db": snr_val,
                    "duration_s": round(duration, 3),
                    "sample_rate": sr,
                    "audiogram_profile": "none", # Will be applied later
                    "split": "test"
                })
    return rows

def build_voicebank_metadata():
    print("Building VoiceBank-DEMAND metadata...")
    vb_dir = os.path.join(RAW_DIR, "voicebank")
    
    train_noisy_dir = os.path.join(vb_dir, "noisy_trainset_28spk_wav")
    train_clean_dir = os.path.join(vb_dir, "clean_trainset_28spk_wav")
    test_noisy_dir = os.path.join(vb_dir, "noisy_testset_wav")
    test_clean_dir = os.path.join(vb_dir, "clean_testset_wav")
    
    rows = []
    
    # Process Train (We'll split some for validation later or keep it all as train)
    if os.path.exists(train_noisy_dir) and os.path.exists(train_clean_dir):
        noisy_files = sorted(os.listdir(train_noisy_dir))
        for idx, wav_file in enumerate(noisy_files):
            if not wav_file.endswith(".wav") or wav_file.startswith("._"): continue
            noisy_path = os.path.join(train_noisy_dir, wav_file)
            clean_path = os.path.join(train_clean_dir, wav_file) # Names match in VB
            
            if not os.path.exists(clean_path): continue
                
            duration, sr = get_audio_info(noisy_path)
            
            rel_noisy = os.path.relpath(noisy_path, RAW_DIR)
            rel_clean = os.path.relpath(clean_path, RAW_DIR)
            
            split = "val" if idx % 10 == 0 else "train" # 10% validation split
            
            rows.append({
                "filename": rel_noisy,
                "clean_file": rel_clean,
                "noise_type": "unknown", # VB doesn't encode this cleanly in filename
                "snr_db": "unknown",
                "duration_s": round(duration, 3),
                "sample_rate": sr,
                "audiogram_profile": "none",
                "split": split
            })

    # Process Test
    if os.path.exists(test_noisy_dir) and os.path.exists(test_clean_dir):
        noisy_files = sorted(os.listdir(test_noisy_dir))
        for wav_file in noisy_files:
            if not wav_file.endswith(".wav") or wav_file.startswith("._"): continue
            noisy_path = os.path.join(test_noisy_dir, wav_file)
            clean_path = os.path.join(test_clean_dir, wav_file)
            
            if not os.path.exists(clean_path): continue
                
            duration, sr = get_audio_info(noisy_path)
            
            rel_noisy = os.path.relpath(noisy_path, RAW_DIR)
            rel_clean = os.path.relpath(clean_path, RAW_DIR)
            
            rows.append({
                "filename": rel_noisy,
                "clean_file": rel_clean,
                "noise_type": "unknown",
                "snr_db": "unknown",
                "duration_s": round(duration, 3),
                "sample_rate": sr,
                "audiogram_profile": "none",
                "split": "test"
            })
            
    return rows

def write_csv(filename, rows):
    path = os.path.join(META_DIR, filename)
    fieldnames = ["filename", "clean_file", "noise_type", "snr_db", "duration_s", "sample_rate", "audiogram_profile", "split"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✅ Saved: {path} ({len(rows)} files)")

def main():
    print("="*50)
    print(" PHASE 2: BUILDING METADATA MANIFESTS")
    print("="*50)
    
    noizeus_rows = build_noizeus_metadata()
    vb_rows = build_voicebank_metadata()
    
    all_rows = noizeus_rows + vb_rows
    
    train_rows = [r for r in all_rows if r["split"] == "train"]
    val_rows = [r for r in all_rows if r["split"] == "val"]
    test_rows = [r for r in all_rows if r["split"] == "test"]
    
    write_csv("train_manifest.csv", train_rows)
    write_csv("val_manifest.csv", val_rows)
    write_csv("test_manifest.csv", test_rows)
    
    print("\nSummary:")
    print(f"  Train : {len(train_rows)} files")
    print(f"  Val   : {len(val_rows)} files")
    print(f"  Test  : {len(test_rows)} files")
    print("="*50)

if __name__ == "__main__":
    main()
