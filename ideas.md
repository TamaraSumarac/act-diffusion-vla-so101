# Ideas

- **Diffusion re-eval via GPU inference** (2026-08-24, after exclusion): community evidence says DDIM-10 runs ~20 Hz on RTX-3090-class GPUs (vs 861 ms/chunk on mps) — fast re-planning dissolves the seam problem. Cheapest path to a Diffusion row on the frozen 20-cell grid: remote policy server or cloud GPU driving the arm; zero model changes. Ranked above ensembling/RTC in the Phase 2.5 options.
- **Targeted-demo probe: data coverage vs architecture** (2026-08-19, after ACT failure taxonomy):
  ACT fails pre-grasp at off-center/90° cells by reaching toward the training mode ("short leash" —
  retries track the block, but only near the workspace middle). Probe: record ~20 extra demos
  concentrated at failing cells (corners, 90°), retrain same ACT, rerun frozen grid.
  If top-right/90° recover → coverage was the bottleneck, no new architecture needed.
  If not → structure/architecture becomes a motivated hypothesis.
  Referenced from results_act.md Interpretation.
- **Structured two-skill policy instead of (or on top of) CVAE** (2026-08-18, after ACT nominal eval 9/20):
  decompose pick-and-place into the two skills that actually decide success —
  (1) position gripper center of mass above the block, (2) orient gripper parallel to block edges.
  Keep action chunking + temporal ensembling; make these two quantities explicit in the model instead of implicit in pixels.
  Cheap probe first: ~20 extra demos at failing cells, retrain same ACT — if top-right/90° recover, it was data, not architecture.
- **Learned skill basis instead of hand-defined skills** (2026-08-18, brainstorm):
  don't define the two skills myself — have a model discover "orthonormal skills" from data
  (structured/discrete latent over trajectory segments, vs. ACT's single global z).
  Possibly discover the decomposition in SIM (structure transfers even when dynamics don't),
  ground it on real demos. Honest eval = recombination/transfer to a new task (table-wiping).
  Read first: DIAYN, VQ-BeT, hierarchical IL segmentation.
