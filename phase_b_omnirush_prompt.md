# OmniRush Prompt — Phase 5B HASPI-Aware Personalization

Continue the audiogram-personalized speech-enhancement project from the
repository root. This is the next step after the accepted Phase 5A foundation.

## Required context

Read these files before changing anything:

```text
AGENTS.md
docs/project_overview.md
context.md
shared_context.md
omnirush_context.md
directives/05_unet_film_model_v2.md
directives/06_evaluation.md
docs/phase_5a_verification_report.md
```

Pull the latest repository state and preserve any local work:

```bash
git pull --ff-only origin main
git rev-parse --short HEAD
git status --short
```

The checkout must contain commit `d20f066` or a descendant. Never commit
checkpoints, raw audio, `.pt`, `.pth`, or `.ckpt` files.

## Phase A decision: do not retrain now

Do **not** retrain Phase A merely because of the latest provenance fix.

The Phase A model, weights, architecture, training objective, and metrics did
not change. The latest fix only makes evaluation CSV line endings and recorded
SHA-256 hashes stable across Windows and POSIX checkouts.

On the RTX 4090 workstation, first run the corrected evaluator once using the
existing checkpoint. This refreshes the JSON/report provenance under the
latest evaluator source without spending another Phase A training run:

```powershell
python execution/phase5_unet_film/26_eval_phase5a.py `
  --data_dir data/processed `
  --checkpoint results/phase5_unet_film/checkpoints/best_composite.pt `
  --results_json results/phase5_unet_film/phase5a_eval_results.json `
  --results_csv results/phase5_unet_film/phase5a_eval_per_sample.csv `
  --manifest results/phase5_unet_film/phase5a_eval_manifest.csv `
  --seed 0

python execution/phase5_unet_film/27_generate_phase5a_report.py `
  --results_json results/phase5_unet_film/phase5a_eval_results.json `
  --report docs/phase_5a_verification_report.md
```

Confirm all of the following before Phase B:

- `pair_count == 824`.
- Per-sample row count is `2472`.
- Manifest and per-sample hashes verify.
- Checkpoint hash verifies.
- Profiles A/B/C are present.
- `mask_invariant_passed == true`.
- `model_is_causal == false` remains explicit.
- No HASPI/HASQI or causal-latency claim is inserted into the Phase A report.

## Phase B objective

Implement a real, validated audiogram-aware personalization stage starting from:

```text
results/phase5_unet_film/checkpoints/best_composite.pt
```

The scientific goal is not merely better generic SI-SDR. The goal is to show
that different audiograms produce useful, measurable differences in hearing-
specific outcomes:

```text
ΔHASPI(Profile B) > ΔHASPI(Profile A)
ΔHASPI(FiLM) > ΔSTOI(FiLM)
```

The Phase A result is currently generic in behavior: Profile A/B/C output
differences are only about `3e-05` to `6e-05`. Treat that as the baseline,
not as successful personalization.

## Non-negotiable safeguards

1. **Never use a random, untrained, or frozen-from-random HASPI network.**
2. **Do not claim HASPI improvement without true HASPI computation** using the
   project’s validated `pyclarity`/MSBG path or an explicitly documented,
   independently checked equivalent.
3. **Do not use the test split for surrogate training.** Keep the official
   VoiceBank test pairs held out for final evaluation.
4. **Do not change the accepted Phase A checkpoint in place.** Write all Phase B
   checkpoints under a separate directory, for example:

   ```text
   results/phase5_unet_film/phase_b/checkpoints/
   ```

5. **Do not implement B1/B2 as placeholders.** If the HASPI dependency or
   training labels are unavailable, stop at a documented blocker and create a
   reproducible data/metric plan instead of fabricating a surrogate.
6. **Preserve the suppression-only invariant** `|M| <= 1`.
7. **Keep Phase A offline semantics.** Do not call the Phase A model causal or
   report streaming latency.

## Recommended implementation order

### Step 1 — Audit available HASPI tooling

Determine whether the workstation environment has a working, documented
HASPI/HASQI implementation. Test it on a tiny deterministic set before writing
the training loop. Record package versions and the exact API.

If `pyclarity` cannot run reliably in the current environment, do not silently
replace it with STOI or a random neural proxy. Document the blocker and stop
before Phase B training.

### Step 2 — Build a real surrogate dataset

Create a deterministic synthetic training/validation split from training data
only. For each example, compute true HASPI targets for multiple audiograms and
store:

- noisy waveform or enhancement input;
- clean reference;
- audiogram vector;
- true HASPI target;
- true HASQI target when available;
- deterministic sample identifier and provenance.

Train a differentiable surrogate only if it achieves a documented held-out
validation target, including R², MAE, and calibration/range checks. The
surrogate must receive the enhanced waveform, clean reference, and audiogram,
not just the waveform and an arbitrary profile embedding.

### Step 3 — Implement B1 as a separate fine-tuning experiment

Create a clearly named script such as:

```text
execution/phase5_unet_film/23_train_phase_b1.py
```

Initialize from `best_composite.pt`, use a lower learning rate than Phase A,
and retain a composite reconstruction term so the model cannot optimize a
perceptual proxy while damaging speech quality.

Save complete metadata for every candidate:

- parent checkpoint hash;
- source Git commit;
- surrogate checkpoint hash;
- training/validation sample IDs;
- audiogram profile distribution;
- all loss weights;
- CUDA/PyTorch/GPU environment;
- finite-loss and finite-gradient status.

### Step 4 — Evaluate the B1 candidate against Phase A

On a fixed validation subset, compare Phase A and B1 using:

- true HASPI for Profiles A/B/C;
- true HASQI for Profiles A/B/C;
- SI-SDR;
- STOI;
- profile-to-profile output differences;
- mask invariant;
- finite-output checks.

Use the directive gate as a starting point, but do not keep a candidate if it
improves HASPI by damaging HASQI or causing instability. Report the actual
measured deltas rather than expected values.

### Step 5 — Only consider B2 after B1 is valid

Implement a real MetricGAN+/LSGAN discriminator only after B1 has a validated
HASPI signal. B2 must include spectral normalization/R1 or an equivalent
stability mechanism, explicit discriminator/generator losses, finite checks,
and a revert path to the accepted Phase A checkpoint.

Do not spend the RTX 4090 training time on B2 if the HASPI evaluator or B1
surrogate has not passed validation.

## Required outputs

At minimum, produce:

```text
execution/phase5_unet_film/23_train_phase_b1.py
execution/phase5_unet_film/29_eval_phase5b.py
results/phase5_unet_film/phase_b/
docs/phase_5b_verification_report.md
```

The Phase 5B report must clearly state whether personalization was detected,
which profile benefited, whether HASQI was preserved, and whether the candidate
was kept or reverted.

## Verification and handoff

Before reporting completion:

```bash
python -m py_compile execution/phase5_unet_film/*.py
python execution/phase5_unet_film/28_verify_phase5a_protocol.py
git diff --check
git status --short
```

Update `context.md` and the Jwanil section of `shared_context.md` with actual
results, blockers, artifact hashes, and the exact next command. Do not claim
Phase B is complete until the true HASPI/HASQI evaluation and the ablation
comparison pass.
