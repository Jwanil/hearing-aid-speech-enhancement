# Directive 00 — Audio/DSP Fundamentals

**Phase:** 0  
**Goal:** Build enough audio signal processing intuition to understand what your model's input/output actually is.  
**Estimated time:** 1 week  
**Output:** Working STFT visualization script + solid conceptual understanding of spectrograms.

---

## What to Learn

1. **Sampling rate** — what 16kHz means, why speech processing uses it
2. **Waveform** — amplitude over time, what it looks like
3. **Fourier Transform** — converting time domain → frequency domain
4. **STFT** — Short-Time Fourier Transform: windowed Fourier over time
5. **Spectrogram** — the 2D image your model will see
6. **Magnitude vs phase** — why you mostly work with magnitude
7. **ISTFT** — going back from spectrogram to audio
8. **Mel scale** — perceptual frequency warping (useful context, not critical yet)

---

## Resources (in order)

1. **Watch first:** 3Blue1Brown — "But what is the Fourier Transform?" (YouTube, ~20 min)
2. **Watch second:** Valerio Velardo — "The Sound of AI" Episodes 1–8 (YouTube)
3. **Reference:** `torchaudio.transforms.Spectrogram` documentation
4. **Reference:** `docs/everything_from_scratch.md` — Parts 1 and 2

---

## Execution Scripts

| Script | What it does |
|--------|-------------|
| `execution/01_stft_visualize.py` | Load a .wav file, plot waveform + spectrogram, play with STFT params |

---

## Tasks

- [ ] Watch 3Blue1Brown Fourier Transform video
- [ ] Watch Velardo episodes 1–8
- [ ] Install all dependencies (`execution/00_verify_setup.py` passes)
- [ ] Write or run `execution/01_stft_visualize.py`
- [ ] Load a speech sample, plot its spectrogram
- [ ] Zero out high-frequency bins and ISTFT back — hear what high-frequency loss sounds like
- [ ] Understand: what does a voiced sound look like vs a fricative on a spectrogram?

---

## Success Criteria

You can explain to your partner:
- What a spectrogram is and how it's computed
- Why you work with spectrograms instead of raw waveforms for this task
- What happens to speech intelligibility when high-frequency bins are removed

---

## Learnings Log

*(Agent: append findings here as you work through this phase)*
