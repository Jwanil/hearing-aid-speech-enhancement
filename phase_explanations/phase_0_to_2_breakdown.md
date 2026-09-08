# Phase 0 → Phase 2: Complete Deep-Dive Explanation

This document explains everything we built across Phases 0, 1, and 2 in thorough detail.
It covers every script's logic, every output file, every graph, and how everything connects to the final goal.

---

## 🎧 Essential Audio & ML Terminology

Read this section first. All of these terms will appear throughout the explanation.

| Term | What it means | Why it matters to us |
|------|--------------|----------------------|
| **Sample Rate (SR)** | Number of audio "snapshots" per second. 16,000 Hz = 16,000 slices per second. | All our models require exactly 16,000 Hz. Mismatched SR = garbage input. |
| **Waveform** | The raw audio signal — a 1D array of numbers from -1.0 to +1.0 representing air pressure over time. | This is what `.wav` files store. Neural nets can work directly on this (Conv-TasNet, Mamba). |
| **STFT** | Short-Time Fourier Transform. Converts a 1D waveform into a 2D time-frequency grid. | Most of our models use STFT. It's the bridge between time and frequency. |
| **Spectrogram** | The visual output of STFT — a 2D heat map (time × frequency × energy). | This is what the U-Net + Mamba models "see" as input. They treat it like an image. |
| **Frequency Bins** | The rows of a spectrogram. Each row = one band of pitch (e.g., 0–31 Hz, 31–62 Hz, ...). | With n_fft=512 at 16kHz, we get 257 bins covering 0–8000 Hz. |
| **dB (decibels)** | Logarithmic measure of loudness. 0 dB = full scale. -80 dB ≈ silence. | We use dB-scale spectrograms because humans hear in log scale, not linear. |
| **dB HL (Hearing Level)** | Medical scale for measuring hearing loss. 0 = perfect hearing, 90 = severe loss. | Audiograms use this scale. Our model receives this as its "personalization" input. |
| **Peak normalization** | Scaling a waveform so its loudest point is exactly ±1.0. | Required before feeding audio to any neural network to prevent exploding gradients. |
| **Mono** | Single audio channel. Stereo has two (left/right). | Neural nets expect one channel. Mixing stereo to mono = averaging both channels. |
| **SNR (Signal-to-Noise Ratio)** | Ratio of clean speech power to noise power, in dB. 0 dB = equally loud. 15 dB = speech dominates. | NOIZEUS gives us 4 SNR levels per noise type. Our models train on low SNR, evaluate on all. |
| **Tensor** | PyTorch's name for a multi-dimensional array (like a numpy array but on GPU). | Every input to our neural nets is a tensor. Shape `[batch, channels, samples]`. |
| **DataLoader** | PyTorch utility that feeds data to the model during training in batches. | Without a DataLoader, you'd hand-craft every batch. It handles parallelism automatically. |

---

## 🔵 PHASE 0: DSP Fundamentals

**Goal:** Prove our environment works, understand how audio is represented mathematically, and build the exact STFT pipeline that our Phase 4/5 models will use as input.

**Phase deliverables:** 2 scripts, 3 plots, 2 audio demo files.

---

### Script 1: `execution/00_verify_setup.py`

#### What it does
Runs 11 import checks across every Python library we'll need across all 7 phases. Prints a clear ✅/❌ report.

#### Code logic
```python
def check(name, fn):
    try:
        fn()
        checks.append((name, True, ""))  # success
    except Exception as e:
        checks.append((name, False, str(e)))  # failure
```
It calls a tiny lambda function that tries to import each library. If the import throws an error (library not installed), it marks that check as failed. At the end it prints the results.

#### Libraries checked and why each matters
| Library | Phase where it's used | Why |
|---|---|---|
| `torch` | 3,4,5 | PyTorch — the deep learning framework. All models run on it. |
| `torchaudio` | 0,1,2,3,4,5 | Audio loading, STFT, resampling. Used in every single script. |
| `numpy` | 0,1,2,3 | Math arrays. MSBG simulator and classical filters run on numpy. |
| `scipy` | 3 | Signal processing. Wavelet thresholding uses scipy. |
| `matplotlib` | 0,1,2,6 | All plotting. Every graph in `results/plots/` is made with this. |
| `librosa` | 3,6 | Advanced audio analysis. Used for PESQ/STOI evaluation metrics. |
| `soundfile` | 0,1,2 | Writing `.wav` files. torchaudio loads, soundfile saves. |
| `pywt` | 3 | PyWavelets — the Discrete Wavelet Transform library for Phase 3 classical baseline. |
| `pystoi` | 6 | Computes STOI (intelligibility metric). Used in final evaluation. |
| `torchmetrics` | 6 | Computes PESQ (perceptual quality metric). Used in final evaluation. |
| `speechbrain` | 4,5 | Pre-trained models, feature extractors. Used to bootstrap our CNN. |
| `pyclarity` | 1 | Cambridge MSBG hearing loss simulator — critical for Phase 1. |

#### When to run this
Every time you clone the project on a new machine. Never skip this — a single missing library will crash a script hours into training.

---

### Script 2: `execution/01_stft_visualize.py`

#### What it does
This script is the most important DSP learning tool in the project. It takes a real human voice recording from our NOIZEUS dataset and:
1. Plots the raw waveform
2. Computes the STFT (Short-Time Fourier Transform) and plots the spectrogram
3. Removes all frequencies above 3000 Hz (simulating hearing loss), converts it back to audio
4. Plots the before/after frequency spectrum

#### Key parameters (lines 54–60)
```python
N_FFT = 512       # 512-sample analysis window = 32ms at 16kHz
HOP_LENGTH = 128  # Step 128 samples between windows = 8ms at 16kHz
WIN_LENGTH = 512  # Same as N_FFT — the Hann window length
TARGET_SR = 16_000
HF_CUTOFF_HZ = 3_000  # Everything above 3kHz is removed
```

**Why these exact values?** These are the industry-standard STFT parameters for 16kHz speech:
- `N_FFT=512` gives us 257 frequency bins, each resolving 31.25 Hz of bandwidth (16000 / 512). Fine enough to see consonant sounds.
- `HOP_LENGTH=128` gives us an 8ms time step — fine enough to follow the rapid changes in speech.
- `WIN_LENGTH=512` = 32ms windows — long enough to capture the periodic structure of vowels, short enough to not blur fast transitions.

#### How STFT works (the `compute_stft` function, lines 90–102)
```python
stft_fn = T.Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH, power=None)
complex_spec = stft_fn(waveform.squeeze(0))  # Output shape: (257, T)
magnitude = complex_spec.abs()
phase = complex_spec.angle()
```
STFT slides a 32ms window across the audio. At each position it applies a Fourier Transform — which decomposes that 32ms slice into its individual frequency components (like a prism splits light). The output is a complex number grid: 257 rows (frequencies) × T columns (time frames).

- `magnitude` = how loud each frequency is at each time moment. This is what we plot and what the neural net uses.
- `phase` = the timing relationship of the frequencies. Critical for converting back to audio. The model predicts masks on the magnitude, then uses the original phase to reconstruct audio.

#### How high-frequency removal works (lines 177–213)
```python
# Forward STFT → (F, T) complex grid
complex_spec = torch.stft(waveform, n_fft=N_FFT, ...)

# Kill all rows above the cutoff bin
cutoff_bin = int(np.round(3000 * 512 / 16000))  # = bin 96 out of 257
filtered = complex_spec.clone()
filtered[cutoff_bin:, :] = 0  # Set rows 96-257 to zero

# Inverse STFT → back to waveform
waveform_out = torch.istft(filtered, n_fft=N_FFT, ...)
```
This is the mathematical equivalent of hearing loss. We convert to frequency space, zero out the high-frequency components (rows 96–257, representing 3kHz–8kHz), and convert back to audio. The result sounds muffled — exactly how a person with high-frequency hearing loss hears speech. **This is precisely what our models will learn to reverse.**

---

#### Output Files: `results/plots/`

##### `00_waveform.png`
- **What you see:** A time-domain plot. X-axis = time in seconds (0 to 4s). Y-axis = amplitude from -1.0 to +1.0.
- **What to look for:** Dense oscillating bursts where the person is speaking (vowels + consonants). Flat sections near 0.0 are silences or pauses. The shape resembles a heartbeat monitor — peaks where sound energy occurs.
- **Why we made it:** Confirms the audio loaded correctly and gives us a time-domain baseline. All our models will eventually output a waveform that should look similar to the clean version of this.

##### `00_spectrogram.png`
- **What you see:** A 2D heat map. X-axis = time (0 to 4s). Y-axis = frequency (0 Hz at bottom, 8000 Hz at top). Color = energy level in dB (yellow/orange = loud, dark purple/black = quiet/silent).
- **What to look for:**
  - Horizontal bands near the bottom (0–1000 Hz) = the fundamental pitch and vowels ("oh", "ah"). These are always present during speech.
  - Diagonal striping patterns in the mid-range (1000–3000 Hz) = formants (vocal tract resonances that distinguish vowels from each other).
  - Scattered energy in the high range (3000–8000 Hz) = fricatives ("s", "sh", "f", "th"). These are the consonants most damaged by hearing loss.
  - Black gaps between bright patches = silences between words.
- **Why this is important:** This is the exact format our U-Net and Mamba models will operate on. The spectrogram is the "image" the model sees. Its job is to learn a filter (called a Complex Ratio Mask) that amplifies the speech and suppresses the noise in this 2D image.

##### `00_hf_removal_spectrum.png`
- **What you see:** A line graph. X-axis = frequency (0 to 8000 Hz). Y-axis = average energy at each frequency in dB. Two lines: blue = original, red dashed = after removing frequencies above 3kHz. An orange dotted vertical line at 3000 Hz marks the cutoff.
- **What to look for:** The blue and red lines overlap perfectly from 0–3000 Hz. At 3000 Hz the red line drops sharply to -160 dB (mathematically zero). Everything to the right of the orange line is gone.
- **Why this matters:** This plot is your proof that the forward + inverse STFT pipeline works perfectly. When we set bins to zero and reconstruct the audio, the frequency content is faithfully removed. This pipeline (STFT → mask → ISTFT) is exactly what Phase 4 (U-Net) and Phase 5 (Mamba) will use, except those models will learn a "smart" mask from data instead of our crude all-or-nothing cutoff.

---

## 🟠 PHASE 1: Audiology + Audiogram Simulation

**Goal:** Build the personalization layer. Generate clinical hearing profiles (audiograms) and use a medical-grade simulator to produce examples of what speech sounds like through damaged ears. The audiogram vector is the unique contribution of this project — it's what separates our personalized models from generic speech enhancement.

**Phase deliverables:** 2 scripts, 4 plots, 5 audio demo files, 1 JSON data file.

---

### Script 3: `execution/02_audiogram_generator.py`

#### What it does
Generates and saves two types of audiograms:
1. **3 fixed test profiles** — used consistently for evaluating all 5 models
2. **8 random realistic audiograms** — used during training to expose the models to diverse hearing loss patterns

#### What an audiogram is
An audiogram is a graph plotted by an audiologist. The patient wears headphones and presses a button when they can barely hear a tone. The softest tone they can detect at each frequency is their "threshold." It's plotted as dB HL (Hearing Level) vs. frequency (Hz).

**Key inversion:** Audiogram graphs are plotted upside-down. 0 dB HL (perfect hearing) is at the TOP. 90 dB HL (severe loss) is at the BOTTOM. This is a medical convention — "higher" on the graph actually means "better" hearing.

#### The 3 fixed test profiles (lines 60–67)
```python
TEST_PROFILES = {
    "Mild Flat Loss":                [15, 20, 25, 25, 30, 30],
    "Moderate Sloping Loss":         [10, 15, 30, 50, 65, 80],  # Presbycusis
    "Severe High-Frequency Loss":    [10, 10, 35, 65, 85, 90],
}
```
These numbers represent the hearing threshold in dB HL at 6 frequencies: [250Hz, 500Hz, 1000Hz, 2000Hz, 4000Hz, 8000Hz].

- **Mild Flat:** Uniformly slightly elevated thresholds. The person hears everything, just a bit softer. The `[15, 20, 25, 25, 30, 30]` means they need sounds 15–30 dB louder than normal to hear them.
- **Moderate Sloping (Presbycusis):** `[10, 15, 30, 50, 65, 80]` — near-normal at low frequencies, severe at high. This is classic age-related hearing loss. The person hears vowels fine ("oh", "ah") but misses consonants ("s", "sh", "f", "th"). This is the most clinically common profile.
- **Severe HF:** `[10, 10, 35, 65, 85, 90]` — catastrophic loss above 2000 Hz. Speech becomes almost completely unintelligible for this person.

These 3 are fixed and identical across all 5 model evaluations so results are directly comparable.

#### Random audiogram generation (lines 84–114)
```python
low_thresh  = np.clip(rng.normal(20, 10), 0, 40)   # 250Hz threshold
high_thresh = np.clip(rng.normal(65, 20), low_thresh, 100)  # 8000Hz threshold
# Log-frequency interpolation between low and high
log_freqs = np.log2(FREQUENCIES)
interp = (log_freqs - log_freqs[0]) / (log_freqs[-1] - log_freqs[0])
base = low_thresh + interp * (high_thresh - low_thresh)
# Add correlated jitter (real hearing loss isn't perfectly smooth)
jitter = np.cumsum(rng.normal(0, 3, 6))
levels = np.clip(base + jitter, 0, 110)
```
**Why this logic?** Real clinical audiograms aren't random noise — they follow patterns:
1. Low frequencies (250–500 Hz) are almost always better than high frequencies. This is a physical fact of cochlear anatomy.
2. The shape tends to "slope" downward from left to right (low→high frequency).
3. Adjacent frequencies are correlated — if 2000 Hz is bad, 4000 Hz is probably also bad.

The `np.cumsum(rng.normal(0, 3))` for jitter models this correlation by taking a random walk — each step is small and builds on the previous one, preventing wild jumps between adjacent frequencies.

**Why log-frequency interpolation?** Human pitch perception is logarithmic, not linear. The perceptual gap between 250 Hz and 500 Hz is the same as between 4000 Hz and 8000 Hz (one octave each). `np.log2(FREQUENCIES)` maps the frequencies onto this perceptually uniform scale before interpolating.

#### JSON output: `results/data/audiograms.json`
This JSON is read by the DataLoader (`03c_data_pipeline.py`) and eventually by the FiLM conditioning layers in the U-Net and Mamba models (Phases 4 and 5). Every time the DataLoader fetches a training sample, it randomly picks one of the 11 audiograms from this file and attaches it to the clean/noisy pair.

---

#### Output Files: `results/plots/`

##### `01_audiograms_profiles.png`
- **What you see:** A single audiogram chart with 3 lines (blue circles, red squares, green triangles). X-axis = frequency on a log scale (250–8000 Hz). Y-axis = hearing threshold level (dB HL), plotted inverted (0 at top, 120 at bottom). Horizontal dashed lines and colored bands mark severity zones (Normal, Mild, Moderate, Mod-Severe, Severe, Profound).
- **What to look for:**
  - The blue "Mild Flat" line stays in the yellow "Mild" band horizontally — it barely changes shape across frequencies.
  - The red "Moderate Sloping" line drops steeply from left to right — barely yellow at 250Hz, dipping into dark red/purple at 8000Hz (80 dB HL). This is the classic presbycusis shape.
  - The green "Severe HF" line starts flat then plunges catastrophically after 1000 Hz, reaching 85–90 dB HL (Severe/Profound zone) at the high frequencies.
  - Numbers annotated on each point show the exact dB HL threshold.
- **Clinical interpretation:** For a moderate sloping loss patient, the threshold at 8000 Hz is 80 dB HL — meaning a sound at 8000 Hz has to be played at 80 dB louder than normal before they can hear it. That's like having to shout to be heard in what should be a quiet room.

##### `01_audiograms_random.png`
- **What you see:** A grid of 8 individual audiogram charts (2 rows × 4 columns), each titled with a severity classification. Each shows a single blue line plotted on the standard inverted audiogram format.
- **What to look for:** The diversity of shapes — some are steep slopes, some are flat, some have a notch (dip) at 4000 Hz (classic noise-induced hearing loss from machinery/concerts). The severity label shows "Mild", "Moderate", or "Mod-Severe."
- **Why we need these 8 random ones:** If we only trained our model on the 3 fixed profiles, it would only learn those 3 specific patterns and would fail on any real-world patient with a slightly different audiogram. The 8 random profiles teach the model to generalize to any shape of hearing loss curve. Combined with the 3 fixed ones, the DataLoader has 11 audiogram profiles to randomly assign during training.

---

### Script 4: `execution/03_simulate_hearing_loss.py`

#### What it does
Takes the real human voice recording from the dataset and passes it through the Cambridge MSBG (Moore-Sek-Baer-Glasberg) hearing loss simulator from the `pyclarity` library. Produces 3 degraded audio files (one per profile) and a 4-panel spectrogram comparison plot.

#### Why the MSBG model (and not just zeroing out frequencies)
The crude STFT zeroing we did in Phase 0 is a toy demonstration. A real cochlea (inner ear) does far more than a simple frequency cutoff:
- **Outer hair cell (OHC) damage:** Reduces the active amplification of soft sounds. The ear loses its ability to amplify quiet frequencies selectively.
- **Inner hair cell (IHC) damage:** Reduces the conversion of mechanical vibrations to neural signals.
- **Broadening of frequency tuning curves:** Damaged hair cells can't distinguish frequencies as precisely. A tone at 4000 Hz causes responses at 3500 Hz and 4500 Hz too.
- **Loss of loudness recruitment:** Soft sounds are inaudible, but loud sounds can be painfully loud.

MSBG models all of these effects simultaneously using physiological parameters derived from clinical data.

#### The technical challenge we solved (pyclarity API)
The MSBG model requires the audio at exactly **44,100 Hz** (not our standard 16,000 Hz). So the script does:
```python
# Load the 16kHz processed file from Phase 2
waveform, sr = torchaudio.load(wav_path)  # sr = 16000

# Upsample to 44100 Hz for MSBG
resampler = T.Resample(16000, 44100)
waveform = resampler(waveform)

# Run MSBG
audiogram = Audiogram(levels=levels, frequencies=FREQUENCIES)
ear = Ear(sample_rate=44100.0)  # Must be float!
ear.set_audiogram(audiogram)
result = ear.process(signal)  # Returns list of arrays
```
The correct classes are `Audiogram` from `clarity.utils.audiogram` and `Ear` from `clarity.evaluator.msbg.msbg` — this was discovered through debugging and is now documented in `directives/01_audiology.md`.

---

#### Output Files

##### Audio demos: `results/audio_demos/`
| File | Content | What to listen for |
|---|---|---|
| `01_original_44k.wav` | Clean speech, sp01 from NOIZEUS, resampled to 44.1kHz | Reference — the ground truth |
| `01_mild_flat_loss.wav` | MSBG with [15,20,25,25,30,30] dB HL | Slightly duller, speech still intelligible, all consonants audible |
| `01_moderate_sloping_loss.wav` | MSBG with [10,15,30,50,65,80] dB HL | 's', 'sh', 'th', 'f' sounds become faint or absent. "Speech" sounds like "ee" |
| `01_severe_hf_loss.wav` | MSBG with [10,10,35,65,85,90] dB HL | Very muffled. Consonants completely lost. Speech nearly unintelligible |

**Critical connection to the project:** These `.wav` files prove what we're solving. Our three deep-learning models (1D CNN, U-Net, Mamba) will receive the severe/moderate version as input and must output something as close as possible to the original. The audiogram vector `[10,10,35,65,85,90]` / 120.0 will be the personalization signal — the model will know exactly which frequencies to boost and by how much.

##### `01_hearing_loss_spectrograms.png`
- **What you see:** 4 side-by-side spectrograms with a shared color scale (-80 dB to 0 dB). Left panel = Original. Next three panels = Mild, Moderate, Severe simulations.
- **What to look for — the key insight of the whole project:**
  - In the "Original" panel, you'll see bright energy across the full frequency range, especially the horizontal streaks in the 3000–8000 Hz range (fricatives and sibilants).
  - In "Mild Flat Loss," the spectrogram looks almost identical to original but slightly dimmer overall.
  - In "Moderate Sloping Loss," the top half of the plot (above ~2000 Hz) goes visibly darker. The bright streaks at 4000–8000 Hz have faded significantly. This corresponds exactly to the sloping audiogram — the model attenuated the frequencies proportionally to the hearing loss.
  - In "Severe HF Loss," the top 60% of the spectrogram is almost completely black (near silent). The only visible energy is the low-frequency vowel formants at the bottom. This is the catastrophic case.
- **Connection to the model:** These 4 panels are literally the training pairs. The model sees the rightmost (most damaged) spectrogram as input, receives the audiogram vector `[10,10,35,65,85,90]` as conditioning, and must output a prediction that matches the leftmost (original clean) spectrogram.

---

## 🟢 PHASE 2: Data Pipeline

**Goal:** Download the training datasets, standardize every file to a uniform format, build a metadata index, and create the PyTorch DataLoader that will feed audio + audiograms into the neural network during training.

**Phase deliverables:** 3 scripts, 3 CSV manifests, 25,541 standardized audio files (on SSD), 1 test plot.

---

### Script 5: `execution/03a_build_metadata.py`

#### What it does
Scans the raw dataset folders on the SSD, reads every `.wav` filename, parses the noise type and SNR from the filename, pairs each noisy file with its clean reference, and writes three CSV manifest files.

#### Why a CSV manifest instead of just using the folder directly
Neural networks train through thousands of iterations per second. If the DataLoader had to scan the SSD directory tree every time it needed a file, it would be catastrophically slow. The CSV is loaded once into memory at the start of training — it's an index of everything on disk. The DataLoader reads a row (a file path), loads that one file, and is done. Fast.

#### NOIZEUS parsing (lines 15–60)
```python
for noise_type in os.listdir(noizeus_noisy_dir):        # 'babble', 'car', 'train', ...
    for snr_folder in os.listdir(noise_dir):            # '0dB', '5dB', '10dB', '15dB'
        for wav_file in os.listdir(snr_dir):            # 'sp01_babble_sn0.wav', ...
            if wav_file.startswith("._"): continue      # Skip macOS hidden files
            speaker_id = wav_file.split("_")[0]         # 'sp01'
            clean_path = f"{speaker_id}.wav"            # 'sp01.wav' in clean folder
```
NOIZEUS encodes everything in the filename: `sp01_babble_sn0.wav` = speaker 01, babble noise, 0 dB SNR. The script parses this and writes it all to the CSV so the DataLoader knows the noise type and SNR for every file.

**The macOS hidden file bug:** When you unzip files on a Mac onto an external drive, macOS creates `._filename.wav` "AppleDouble" metadata files alongside each real file. `torchaudio.load("._sp01.wav")` crashes with "Format not recognised". The fix: `if wav_file.startswith("._"): continue`.

#### VoiceBank-DEMAND parsing (lines 62–120)
Unlike NOIZEUS, VoiceBank-DEMAND uses a 10% validation split:
```python
for idx, wav_file in enumerate(noisy_files):
    split = "val" if idx % 10 == 0 else "train"  # Every 10th file = validation
```
This deterministically assigns 10% of the training files to validation (checking model performance during training) and the other 90% to actual training.

#### CSV schema
```
filename, clean_file, noise_type, snr_db, duration_s, sample_rate, audiogram_profile, split
noisy/voicebank/noisy_trainset.../p226_001.wav, clean/voicebank/..., unknown, unknown, 4.0, 16000, none, train
noisy/noizeus/babble/0dB/sp01_babble_sn0.wav, clean/noizeus/clean/sp01.wav, babble, 0, 4.0, 8000, none, test
```

**Final counts:** 9,839 train | 1,093 val | 1,754 test

---

### Script 6: `execution/03b_standardize_audio.py`

#### What it does
Processes all 25,541 raw `.wav` files. For each file: load → convert to mono → resample to 16kHz → normalize amplitude → pad/trim to exactly 4 seconds. Saves to `data/processed/`.

#### Why standardization is non-negotiable
A neural network is a fixed mathematical function. Its input layer expects a tensor of exactly one shape. If you feed it a 3-second file one iteration and a 7-second file the next, the matrix multiplications fail. All 25,541 files must be identical in shape: `(1, 64000)`.

The 4 specific operations, and why:

**1. Mono conversion**
```python
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
```
Some VoiceBank files are stereo (2 channels). Neural nets need 1 channel. Average both channels so no information is lost.

**2. Resampling to 16kHz**
```python
if src_sr != TARGET_SR:
    resampler = torchaudio.transforms.Resample(src_sr, TARGET_SR)
    waveform = resampler(waveform)
```
NOIZEUS files are 8 kHz. VoiceBank and MUSAN are already 16 kHz. MSBG outputs at 44.1 kHz. The Resample transform uses polyphase filtering to correctly interpolate/decimate the signal — not just nearest-neighbour (which would alias). Crucially, 16 kHz captures speech frequencies up to 8 kHz (Nyquist limit), which is all of the medically significant hearing range.

**3. Peak normalization**
```python
peak = waveform.abs().max()
if peak > 0:
    waveform = waveform / peak
```
Scale so the loudest sample is exactly 1.0. This prevents: (a) exploding gradients from loud files overwhelming the model, (b) silent files providing zero useful gradient signal. After normalization, all files are on the same amplitude playing field.

**4. Length standardization to 4 seconds (64,000 samples)**
```python
TARGET_LEN = 64000  # 4 seconds × 16,000 Hz
if length > TARGET_LEN:
    waveform = waveform[:, :TARGET_LEN]        # trim from end
elif length < TARGET_LEN:
    pad = TARGET_LEN - length
    waveform = F.pad(waveform, (0, pad))       # zero-pad at end
```
4 seconds was chosen because: (a) all NOIZEUS utterances are ≤3.5s — 4s covers them with room; (b) VoiceBank utterances range from 1s to 10s — 4s is a reasonable chunk size; (c) 64,000 samples is a power-of-2-friendly size for Conv-TasNet and STFT operations.

**Multiprocessing:**
```python
with multiprocessing.Pool(workers) as pool:
    results = list(tqdm(pool.imap_unordered(process_file, all_files), ...))
```
Uses all CPU cores in parallel. On a typical Mac with 8 cores, 25,541 files take ~60 seconds instead of ~8 minutes sequentially.

#### Result
25,541 files in `data/processed/`, each guaranteed to be shape `(1, 64000)`, 16 kHz, amplitude in [-1, 1].

---

### Script 7: `execution/03c_data_pipeline.py`

#### What it does
Implements the `HearingAidDataset` class (inheriting from PyTorch's `Dataset`) and tests the DataLoader with a batch of 4 samples.

#### The class: `HearingAidDataset`

**`__init__`:** Loads the CSV manifest into memory as a list of dictionaries. Loads `audiograms.json` and builds a lookup table: `{profile_name: [6 dB HL values]}`.

**`__len__`:** Returns the number of rows in the CSV. PyTorch uses this to know when one "epoch" of training is complete.

**`load_audio`:** Given a relative path from the CSV, loads the processed file from `data/processed/` and enforces the exact `(1, 64000)` shape as a safety net (in case standardisation missed a file).

**`__getitem__` — the core logic (lines 71–99):**
```python
def __getitem__(self, idx):
    item = self.items[idx]                    # One CSV row

    noisy = self.load_audio(item["filename"])    # Noisy speech tensor (1, 64000)
    clean = self.load_audio(item["clean_file"])  # Clean speech tensor (1, 64000)

    # Randomly pick one of 11 audiogram profiles
    ag_name = random.choice(self.audiogram_keys)
    ag_thresholds = self.audiograms[ag_name]     # [15, 20, 25, 25, 30, 30]

    # Normalize dB HL range [0, 120] → [0.0, 1.0]
    ag_tensor = torch.tensor(ag_thresholds) / 120.0  # [0.125, 0.167, 0.208, ...]

    return {
        'noisy': noisy,         # Shape: (1, 64000)
        'clean': clean,         # Shape: (1, 64000)
        'audiogram': ag_tensor, # Shape: (6,)
        'snr': snr,             # Float
        'filename': filename    # String (for debugging)
    }
```

**Why normalize audiogram to [0, 1]?** Neural networks work best when all input values are in a consistent, bounded range. A raw dB HL value of 90 is 90× larger than 1 dB HL — this imbalance would distort the gradient updates. Dividing by 120 (the maximum possible value) scales everything to [0, 1] while preserving the relative ratios.

**Why randomly assign audiograms?** At training time, we don't know which hearing profile a given audio file was recorded by a person with. We want the model to learn: "given this noisy audio AND this audiogram, produce this clean audio." By randomly assigning audiograms, we force the model to learn to use the audiogram vector meaningfully rather than ignoring it.

#### DataLoader batch
```python
loader = DataLoader(dataset, batch_size=4, shuffle=True)
batch = next(iter(loader))
# batch['noisy']:    shape (4, 1, 64000) — 4 samples, 1 channel, 64000 timepoints
# batch['clean']:    shape (4, 1, 64000)
# batch['audiogram']:shape (4, 6)        — 4 audiogram vectors, 6 freq thresholds each
```
**batch_size=4** means the GPU processes 4 audio files simultaneously. This is a small number for testing — during actual training it will likely be 8–32.
**shuffle=True** means each epoch presents files in a random order, preventing the model from memorizing patterns based on file order.

---

#### `results/plots/02_dataloader_test.png`
- **What you see:** Two stacked spectrograms from the validation set. Top = clean speech, bottom = noisy speech with the same content. Both are 4 seconds at 16 kHz. Title shows the filename and SNR level.
- **What to look for:**
  - The top (clean) spectrogram should have clean, well-defined speech patterns — clear horizontal bands at vowel formant frequencies, clean bursts at fricative frequencies.
  - The bottom (noisy) spectrogram should show the same speech BUT overlaid with noise. Depending on the noise type (babble, street, car), you'll see either diffuse speckle energy (babble) or a colored noise floor (car/street), both of which partially mask the speech content.
- **Why this plot matters:** It visually confirms that our DataLoader is actually loading the right pairs (noisy and clean are the same utterance, not randomly mismatched) and that the standardization pipeline produced valid audio (not silence or garbage). This is the exact format that will go into Phase 4 model training.

---

## 🗂️ How Everything Connects: The Full Pipeline

```
Phase 0 (STFT pipeline confirmed)
    ↓
Phase 1 (Audiogram profiles + MSBG simulation — understand the problem)
    ↓
Phase 2 (25,541 standardized files + DataLoader → feeds Phases 3–6)
    ↓
Phase 3: Classical Baselines (Wavelet DWT + MMSE-LSA — math filters on STFT)
    ↓
Phase 4: 1D CNN / Conv-TasNet (generic waveform model, no audiogram)
    ↓
Phase 5: U-Net + FiLM / Mamba + FiLM (personalized models — audiogram goes in here)
    ↓
Phase 6: Evaluation (HASPI, STOI, PESQ, DNSMOS on all 5 models, all 3 audiograms)
```

The STFT code from Phase 0 → used directly in Phase 4/5 models.
The audiogram profiles from Phase 1 → used by the FiLM conditioning layer in Phase 5.
The DataLoader from Phase 2 → used as-is in every training loop in Phases 4 and 5.
The NOIZEUS test set from Phase 2 → used as-is in the evaluation in Phase 6.

**Nothing from Phases 0–2 is throwaway.** Every file, every script, every output plugs into the phases that follow.
