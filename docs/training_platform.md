# Training Platform Recommendation

## Recommended: RunPod (Primary) / Vast.ai (Fallback)

### Why RunPod

| Factor | RunPod | Vast.ai | Colab |
|--------|--------|---------|-------|
| **GPU Availability** | RTX 4090 24GB, A100 40/80GB — consistent supply | Similar, but more spot preemption | T4/A100 — limited, frequent disconnects |
| **Storage** | Local NVMe `/tmp` (100GB+ included) — critical for MUSAN+RIR cache (avoids 40% GPU idle on Drive FUSE) | Local NVMe available | Google Drive FUSE only — 80ms/load, GPU starves |
| **Billing** | Per-second, **Community $0.34/h, Secure $0.89/h** for 4090 (verified Oct 11 2026) | Per-second, similar | Free tier unstable; Pro $10/mo but quota-limited |
| **Environment** | `runpod/pytorch:2.4.0-cuda12.4.1` (or 2.2.0) — `mamba_ssm` + `causal-conv1d --no-build-isolation` | Same containers work | `nvcc` often missing → `mamba_ssm` falls back to slow `S4D` |
| **Persistence** | Pod survives 24h+; checkpoint to Drive via `rclone` async | Similar | 12h max, random preemption loses state |

### Required Spec for This Project

| Phase | GPU | VRAM | Disk | Est. Time | Community $0.34/h | Secure $0.89/h |
|-------|-----|------|------|-----------|-------------------|----------------|
| Phase 5 U-Net (Rev 2.1) | RTX 4090 24GB | 8–12GB | 30GB container + 20GB volume | 10.5–14h (Phases A 5.5h + pilots 6h + ext 2.5h) | **$3.57–$4.76** | **$9.66–$12.52** |
| Phase 5b Mamba (Rev 2.0) | RTX 4090 24GB | 10–16GB | 30GB + 20GB | 7–14h (base/S4D) | $2.38–$4.76 | $6.23–$12.46 |

> **Oct 2026 note:** Community RTX 4090 is $0.34/h but frequently sold out. If unavailable, Secure 4090 is $0.89/h — then Phase 5 sequential 14h = $12.52 (tight on $12 credit). Use gated plan: run Phase A (5.5h = $4.92 Secure), then Pilots B1/B2 only if gate warrants, to stay within $12.

### Quick Start (RunPod)

```bash
# 1. Create pod: runpod/pytorch:2.2.0-cuda12.1, 4090 24GB, 100GB container disk, 50GB volume
# 2. SSH in, then:
git clone https://github.com/<your-repo>.git hearing-aid-speech-enhancement
cd hearing-aid-speech-enhancement

# 3. Preload data to local NVMe (critical — avoids Drive FUSE bottleneck)
mkdir -p /tmp/hearing-aid-data
gsutil -m cp -r gs://<your-bucket>/processed/* /tmp/hearing-aid-data/  # or rclone from Drive
# MUSAN+RIR ~10GB, VoiceBank+NOIZEUS+Clarity ~15GB

# 4. Install deps (mamba_ssm needs --no-build-isolation on some images)
pip install -q torch torchaudio pyclarity pystoi pesq accelerate wandb
pip install -q causal-conv1d mamba_ssm --no-build-isolation || echo "S4D fallback active"

# 5. Train Phase 5 U-Net
python execution/phase5_unet_film/21_train_unet_film.py \
  --phase A --epochs 25 --batch 8 --accum 2 --lr 3e-4 \
  --data_dir /tmp/hearing-aid-data \
  --ckpt_dir /tmp/checkpoints/phase5A \
  --mixed_precision bf16

# 6. After Phase A: run Pilot B1/B2 in parallel (separate pods) for conditional GAN gate
```

### Colab + colab-mcp (Development/Prototyping Only)

- Use for: architecture debugging, small-scale test runs (<1 epoch), streaming harness profiling
- **Do not use for full training** — Drive FUSE I/O kills GPU utilization, 12h timeout loses checkpoints, no persistent NVMe
- `colab-mcp` MCP server bridges this IDE to a live Colab notebook for interactive dev
- Setup guide: `docs/colab_setup.md`

### Hardware Targets (Post-Training)

| Tier | Device | Model | Quantization | Latency | Power | When |
|------|--------|-------|--------------|---------|-------|------|
| **A (Nov 1 Demo)** | **Phone (SD 7xx/8 Gen) / Laptop / ESP32-S3 ($30 alt)** | U-Net 770k (phone/laptop FP16) / 310k Tiny (ESP32 INT8) | TFLite GPU/NNAPI / TFLite Micro | 14–24 ms phone wired (6.8 ms algo), 18–26 ms ESP32 | 0.8 W / 70 mW |
| **B** | **Skipped** | — | — | — | — |
| **C (Post-Nov)** | GAP9 NE16 / Ezairo 7150 | Tiny U-Net 310k / Mamba Tiny 950k (GAP9 only) | INT8 (+ INT16 state for Mamba) | 6.87 ms (GAP9) | 11 mW |

See `docs/hardware_deployment_roadmap.md` (Tier A directive + future overview) and `docs/future_plan_tier_c_implementation.md` (full Tier C BOM/bench).

---

**Bottom line:** Provision a **RunPod RTX 4090 24GB with 100GB NVMe**, preload data to `/tmp`, train with `bf16` + `Accelerate`. This is the only configuration that reliably delivers the 10–14h training windows for Phases 5/5b within budget and without I/O bottlenecks.