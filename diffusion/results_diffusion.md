## Deployment note: inference latency and the DDIM-10 checkpoint variant

**Problem.** The off-robot smoke test timed the trained Diffusion Policy checkpoint at
**9.2 s per action chunk** on the deployment machine (M-series Mac, mps). A chunk of 32
actions covers 32/30 ≈ 1.07 s of robot time at 30 Hz, so the policy would need ~9× real
time to keep the robot fed — undeployable (the arm would freeze ~8 s between every chunk).
The cause is architectural: the training preset uses DDPM sampling with
`num_train_timesteps=100`, and with `num_inference_steps=None` inference runs all 100
iterative U-Net denoising passes per chunk (~65 ms per pass for the 263M U-Net on mps).

**Fix (the paper's own recipe).** Diffusion Policy prescribes training with DDPM and
sampling at deployment with DDIM using ~10 steps, at small documented quality cost. The
question was where to apply the override.

**What we tried, gentlest first:**
1. **Mutating the config after loading** (`policy.config.noise_scheduler_type = "DDIM"`,
   `policy.config.num_inference_steps = 10` after `from_pretrained`): no effect on timing.
   The scheduler object is constructed inside `__init__` from the config values at build
   time; editing the config afterwards changes the record, not the machinery.
2. **Passing overrides as `from_pretrained(..., noise_scheduler_type=..., num_inference_steps=...)`
   kwargs:** also ineffective — a verification print of
   `type(policy.diffusion.noise_scheduler).__name__` still showed `DDPMScheduler`, i.e.
   the kwargs did not reach config construction in this LeRobot version.
3. **Editing the checkpoint's `config.json` directly, in a copy:** guaranteed to work,
   because `from_pretrained` must read that file to construct the policy — there is no
   code path around it. The trained checkpoint was copied to
   `checkpoints_diffusion_baseline/100000_ddim10/` and only two fields changed:
   `"noise_scheduler_type": "DDIM"`, `"num_inference_steps": 10`. The original
   checkpoint remains untouched, and the variant's name makes the eval configuration
   self-documenting.

**Result.** Verified `DDIM 10 DDIMScheduler` at load; inference dropped **9.2 s → 861 ms**
per 32-action chunk (10.7×), inside the 1.07 s chunk budget. Prediction accuracy on a
training frame stayed within the same few-degree band as DDPM sampling (run-to-run
variation of 1–2° is expected either way — diffusion sampling is stochastic, unlike ACT's
deterministic inference).

**Caveats carried into the eval.** 861 ms is inside the chunk budget but not seamless:
with synchronous inference the arm pauses ~0.9 s at each re-plan (the strict
receding-horizon budget for the executed portion of each chunk is ~0.27 s). Evaluation
proceeds with visible think-pauses; this does not affect the success criterion, but
chunk-generation cost is recorded as an architecture-level deployment property of the
comparison (ACT: tens of ms per 100-action chunk; Diffusion DDIM-10: ~861 ms per
32-action chunk). All nominal-eval rollouts use the `100000_ddim10` variant.

**Community context.** This failure mode is documented in the field, not specific to
this setup: jerky diffusion motion on SO-101 is reported even on RTX-4080-class
hardware (with denoising-steps and action-horizon tuning failing to help, as here),
and practitioner guidance puts DDIM-10 at ~20 Hz on an RTX 3090 — roughly 5–10×
faster than the 861 ms measured on mps. At GPU-class re-planning rates the
inter-chunk seam problem largely dissolves (fast re-planning is the architecture's
intended smoothing mechanism), and published deployments of this exact LeRobot
implementation on SO-100 at 30 Hz control run on A6000-class servers. The exclusion
is therefore scoped precisely: Diffusion Policy is undeployable on laptop-class
(mps) inference under sync rollout — not undeployable in general. GPU-backed
inference (local or via policy server) is the first-ranked option for a follow-up
(ideas.md).