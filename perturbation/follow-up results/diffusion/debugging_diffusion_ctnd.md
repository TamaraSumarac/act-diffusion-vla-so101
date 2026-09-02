## Debugging diffusion violent motion that damaged the robot (offline action signal analysis)

**Motivation.** The DDIM-10 hardware run was violent enough to detach a sensor. Before
the arm moves on this policy again, the question "where does the violence come from?"
was answered on the desk: run the checkpoint offline on recorded observations and
measure the action waveform, rather than discovering it on hardware. Three candidate
mechanisms, separable by measurement: (a) intra-chunk noise (coarse DDIM sampling →
trembling plans), (b) chunk-boundary seams (consecutive re-plans disagree), (c)
sync stop-go timing (not measurable from action values; the residual if (a) and (b)
come back clean).

**Method.** Notebook (`tools/diffusion_signal_analysis.ipynb`): 30 frames sampled
mid-episode from the frozen training dataset. For each frame t, the policy predicts its
full action chunk from exactly the observations it would have seen live (2-frame history
for Diffusion, built via `delta_timestamps` to mirror training collation). Metrics, all
in degrees:

- *Intra-chunk*: per-step deltas within one predicted chunk (does the plan tremble?).
- *Seam*: predict again from frame t+n (n_exec ∈ {8, 16}) and measure
  |chunk_B[0] − chunk_A[n−1]| — the commanded jump at a re-plan boundary.
- *Resample spread*: same frame sampled K=6 times; std across draws isolates the
  contribution of sampling stochasticity alone.

Schedulers compared on identical frames (same seed → same frame set): DDPM-100
(as trained), DDIM-50, DDIM-10 (the deployed variant), plus ACT as deterministic
reference and dataset ground-truth deltas as the smoothness floor. Scheduler variants
by checkpoint `config.json` surgery with symlinked weights (the only override path
that works; see the DDIM-10 note above). The variant-existence guard now verifies
scheduler/steps content, after a stale symlink was caught silently aliasing "ddim50"
to the original checkpoint.

Caveat: teacher-forced. Frame t+n comes from the demo trajectory, not from executing
chunk A, so seam numbers are a lower bound on live seams (live, the arm has
additionally drifted off the demo path).

**Pipeline notes** (recorded because each cost a debugging round):
- Normalization lives in the checkpoint's pre/post processor pipelines
  (`make_pre_post_processors`), not in the policy. Calling `predict_action_chunk` on
  raw dataset tensors silently produces normalized-space garbage — first-pass numbers
  were invalid until predictions were verified against ground truth in degrees on a
  training frame.
- Diffusion's `predict_action_chunk` unsqueezes a fake single-frame history
  (n_obs_steps=1) — the same design that leaves the async server's obs queues unfilled.
  Correctly stacked history must bypass it and call `diffusion.generate_actions`
  directly, adding the batch dim manually (the preprocessor's ndim check mistakes
  history-stacked state for an already-batched input).

**Results** (30 seam pairs, degrees; dataset floor: p95 = 1.85, max = 5.98 per step):

| policy | intra d_p95 | seam n8 (mean) | seam n16 (mean) | worst seam | resample spread |
|---|---|---|---|---|---|
| ACT | 1.76 | 2.12 | 3.55 | 10.9 | — (deterministic) |
| DDIM-10 | 1.43 | 3.13 | 6.75 | **38.0** | 1.32 |
| DDIM-50 | 1.52 | 3.07 | 3.66 | 21.5 | 1.28 |
| DDPM-100 | 1.47 | 2.93 | 4.31 | 18.5 | 1.08 |

**Findings.**
1. **Intra-chunk noise is exonerated.** Every scheduler's per-step deltas sit at or
   below the dataset floor. The plans do not tremble; hypothesis (a) is dead.
2. **The violence is a seam-tail phenomenon.** Mean seams are benign; the damage lives
   in rare spikes (18–38°, 4–8× the 5° clamp) concentrated at two specific frames —
   visually near-grasp moments where the demo waits and the policy is multimodal about
   *when* to move. Mechanism: at n_exec=16 the executing chunk's open-loop tail has
   drifted far from where a fresh re-plan begins.
3. **More denoising does not fix it.** The spikes persist at the same frames through
   DDIM-50 and DDPM-100 (38 → 21 → 18°), and ACT shares the same hot frames at lower
   magnitude (10.9°). The spikes are a property of the learned model's open-loop drift
   at ambiguous moments, not a DDIM-10 sampling artifact. The "GPU + more steps fixes
   everything" story is dead; the GPU fixes *timing*, not seams.
4. **Executed-horizon length is the lever.** Worst seams at n_exec=8 are 6–9° across
   all schedulers vs 18–38° at n_exec=16. Drift compounds with horizon; re-plan often
   and the seam never grows teeth. (The deployed sync config executed 32 — deeper into
   bad territory than anything measured here.)
5. **Resample overlays** (5 draws, hot frame): DDPM-100/DDIM-50 produce diverse bump
   timings — the multimodality diffusion was chosen for. DDIM-10's draws collapse to a
   tight bundle (known few-step-DDIM cost) yet it has the *worst* seams: low variance
   around a drifting trajectory beats high variance for nothing. All fifteen draws
   across all schedulers also share the same bias (bump ~30 steps before the demo does)
   — model-level, echoing finding 3.

**Deployment consequences.**
- `max_relative_target=5` is load-bearing permanently, not a first-run precaution:
  worst observed commanded jump is 38°, and the spike mechanism is model-level. The
  clamp converts it into a 5°/tick slew. (Demo motion itself occasionally exceeds
  5°/step — p-max 5.98 — so the clamp mildly slows even perfect imitation; acceptable.)
- Target configuration: **remote GPU inference + effective n_exec ≈ 8 + clamp**. Async
  re-plan cadence at A10 chunk times (~0.1s expected for the 263M U-Net) naturally
  lands in the small-n_exec regime, so the timing fix and the seam fix are the same
  fix. A sync clamped run at n_exec=8 on mps would reintroduce the ~0.9s-freeze lurch
  and is skipped.
- Prerequisite: the async server cannot serve Diffusion as-is — `predict_action_chunk`
  fabricates a single-frame history, so the server needs a 2-frame obs buffer
  (the same fix the notebook applies manually).

Artifacts: `results/diffusion_signal/signal_metrics.csv`, `signal_summary.png`,
`resample_overlay_f58.png`, notebook in `tools/`.


## Running inference for diffusion on cloud

**Setup.** Remote inference for Diffusion via the async client/server path (A10 box,
SSH tunnel), extended for diffusion's 2-frame observation history: the client pairs
each sent observation with the frame captured 1/30s earlier (a rolling one-tick
buffer; at episode start the first frame is duplicated, matching sync warm-up), and
the server stacks [prev, cur] before inference, calling `generate_actions` directly —
the stock `predict_action_chunk` fabricates a fake single-frame history and cannot
serve n_obs_steps=2.

**Transport engineering** (each patch forced by a measured failure):
- Raw paired frames = ~1.8MB per send; the synchronous send blocked the 30Hz control
  loop 280–400ms per send. Fix 1: JPEG transport (quality 90, ~120KB) — spikes down
  to 80–180ms, still blocking.
- Fix 2: a background sender thread with a 1-slot newest-wins handoff queue — control
  loop settled at 3–7ms uniformly, paired-frame spacing locked at ~0.033s.
  Both fixes are needed: JPEG keeps the payload inside the chunk budget, the thread
  keeps serialization off the control loop.
- A server-path probe (training frame → jpeg round-trip → server prep → policy,
  compared against the clean offline path and ground truth) agreed to 1.85° max —
  the pipeline was exonerated at every layer, so remaining behavior was the policy's.

**The clamp/sampler knot.** With the pipeline clean, DDIM-10 reproduced the offline
seam findings on hardware, from both directions:
- Unclamped (or clamp 15): travels, violently — the original sensor-breaking regime.
- Clamp 5: zero net travel at every re-plan cadence tried (threshold 0.8 → 0.0).
  The clamp is a low-pass filter on the commanded trajectory; DDIM-10's fast,
  bump-and-return, mode-flipping plans integrate to ~zero through a 5°/tick gate.
  The arm "wiggles in place" — the policy fighting its own re-plans, amplitude-capped.
No commitment length fixes this: even full-chunk sync semantics (threshold 0,
executing all 32 steps from step 0) produced closed loops. The signal itself was the
problem, exactly as the offline resample overlays showed (collapsed single-mode
draws that flip between re-plans).

**What worked: DDIM-50.** Same checkpoint, config.json surgery to 50 steps, ~0.43s
per chunk on the A10 (vs 0.06s DDIM-10, 0.83s DDPM-100). With clamp 10 the arm
smoothly approached and nearly grasped; a disambiguation run (DDIM-10 back at clamp
10) confirmed the sampler, not the clamp, was the active ingredient — a looser gate
admits more signal but cannot make the signal smooth. Clamp bracket: 8 best (moves
up cleanly, minimal jerk), 5 and 10 slightly rougher. Three repeats at the frozen
config: **2/3 pickups, 1 near-miss.**

**Scheduler bracket is non-monotonic.** DDPM-100 at 0.83s/chunk ran flawlessly as
machinery and barely moved the arm (single run, held loosely). Consistent with the
offline overlays: DDPM-100 has the most inter-draw mode variety — full-commitment
execution of net-zero bump-and-return draws plus ~1s pauses accumulates nothing.
DDIM-50 appears to sit at a sweet spot: enough steps to escape DDIM-10's collapsed,
flip-flopping bundle; few enough that draws stay pruned toward a dominant mode.
Deployment-optimal sampling ≠ maximum-fidelity sampling.

**Frozen eval config:** DDIM-50 · `max_relative_target=8` · `chunk_size_threshold=0.0`
(full-chunk sync semantics) · `latest_only` · paired observations · JPEG 90 · remote
A10. ~0.43s think-pause per 1.07s chunk — same duty class as the SmolVLA remote eval.
The clamp is part of Diffusion's deployment requirements (ACT and SmolVLA run
unclamped); that asymmetry is a finding, not a footnote.

**Bottom line.** DDIM-10 was never "the paper's recipe" — it was a compromise forced
by the Mac's 861ms budget. On mps, 10 steps was the only deployable setting and it
was violent; clamping it to safety filtered its plans to zero. The GPU retired the
constraint: 50 steps is deployable at 0.43s/chunk, and the same policy went from
breaking a sensor to picking up the block.
