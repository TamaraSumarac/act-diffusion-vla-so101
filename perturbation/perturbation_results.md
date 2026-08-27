## Pre-registered hypotheses vs outcomes

All hypotheses were written down before any grid trials ran (see perturbation_protocol.md).

### Position shift
Prediction was that as a default both policies might be seeking demo middle, potentially
ACT more than SmolVLA. However, we see that both policies do register that the object is
outside of the box and try to adjust to pick it up. Arguably SmolVLA was seeking the middle
more than ACT actually (7 vs 3 sought_demo_middle failures). Note: adjustment was still not
enough — 1/16 and 0/16 — so the coverage cliff is real, it's just not caused by blind
middle-seeking.

### Novel color
We expected that SmolVLA would be more affected by new color than ACT. This ended up being
true (ACT 45%→55%, SmolVLA 35%→10%). The hypothesis now is that the task string going into
SmolVLA is what's affecting it — "pink block" no longer matches the scene. We can test this
with another measurement where the task string is changed to match the color of the new
target; recovery would confirm instruction grounding as the mechanism.

### Distractor
Hypothesis was that ACT would ignore the sock and SmolVLA would be more distracted, because
the sock is pink and we believe pink is important conditioning for SmolVLA. It is true that
SmolVLA was more distracted. However, it is not true that ACT was unaffected — the
oscillation mode registered for ACT (3 failures, distractor-only) says it is also
registering the distractor. "Doesn't target the sock" and "isn't perturbed by the sock"
turned out to be different claims.

### Lighting
Hypothesis was that ACT would be more affected than SmolVLA, but both policies would get
worse. This is what we saw (ACT 45%→25%, SmolVLA 35%→25%). Consistent with a global pixel
shift hurting ACT's scene-conditioned features more, while pink stays identifiably pink
for SmolVLA.

### Novel object
Hypothesis was that SmolVLA would adjust to a new object better than ACT, given its
pre-trained prior. This is exactly what we saw. ACT's most common failure mode was not
registering the new object at all (no_target_registered, 16/19 failures), whereas
SmolVLA's common failures were near_miss, beside_object, and grasp_slip — all saying it
registers the object and almost succeeded; mainly the contact physics was different and
that's what failed.

### Cube slide
Hypothesis was that ACT would not register the slide as much and would seek the middle of
the demo more, whereas SmolVLA would adjust to the change. This is what we saw. Note the
inversion: ACT replans ~3x faster (~351 ms vs ~1 s) and still didn't re-target — so
representation, not reaction time, is the binding constraint. Also note the success rates
in this column are inflated for both policies (slides moved the block toward
better-covered territory), so the re-tracking claim rests on first-grasp-target coding,
not raw rates.