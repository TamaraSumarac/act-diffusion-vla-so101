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