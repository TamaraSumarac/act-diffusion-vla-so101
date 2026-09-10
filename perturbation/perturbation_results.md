## Key results:

* **Novel color:** Affected SmolVLA more than ACT, suggesting that SmolVLA conditions more strongly on color (consistent with color being explicitly included in its task string).
* **Novel object:** Affected ACT more than SmolVLA. Combined with the novel color result, this suggests that ACT relies more on object shape, while SmolVLA relies more on color and its task string input.
* **Distractors:** Consistent with the color and object findings, SmolVLA degraded more in the presence of a pink distractor than ACT, although ACT was not fully immune to it either.
* **Lighting:** Affected both policies, though SmolVLA was somewhat more robust and had many near-misses (5/15). One possible explanation is its much larger pre-trained visual prior, which was likely exposed to a wider range of lighting conditions during pre-training.
* **Position shift:** Suggests that training data coverage is a major constraint on pickup accuracy. SmolVLA appears to register the object's new location better than ACT, but its "body" acts somewhat "unaware" of how to execute the motion needed to pick up an object in a position it has never seen during training.
* **Object slide:** Further supports the position shift findings. SmolVLA shows better visual adjustment to changes in object position, while ACT operates more on "autopilot," moving its arm toward a familiar location (typically near the middle of the training region). It sometimes succeeds accidentally after pushing, dragging, or tipping the object into that region during previous pickup attempts.

![Robustness heatmap success results](robustness_heatmap.png)

## Results of failure analysis:
If we look at failure modes in aggregate across ACT and SmolVLA, we see some clear behaviors:

* **Shared failure modes:** Both policies consistently make the same three mistakes: `sought_demo_middle`, `tipped_block`, and `pushed_block`. I believe all three are consistent with limitations in training data representation. Tips and pushes often happen when arm is close to object but not exactly there on first try, it accidentally creates contact with the object and moves it toward a more familiar position, while `sought_demo_middle` is an even clearer consequence of the training distribution.

* **ACT — weaker visual registration:** ACT shows more `no_target_registered` and oscillation behavior than SmolVLA, both consistent with ACT's visual model being less capable of detecting and adapting to novel objects/positions. Oscillation happens when ACT registers that there is no object in the starting region and also registeres there is no object in the target tray region, causing it to hover between the two without success (oscillation also occured in distractor measurements when ACT looks like it registered two objects at two distinct locations).

* **SmolVLA — more "almost successes":** SmolVLA has many more `beside_object`, `near_miss`, and `grasp_slip` failures, particularly the latter two. These dominate especially in the novel object experiment, where SmolVLA's visual model appears to work well, but its action model may be less successful at executing the pickup than ACT's more "autopilot" behavior.

* **SmolVLA — wrist orientation:** SmolVLA also shows a unique `wrong_wrist` failure. Its gripper orientation often reverts toward angles closer to 0°, which are better represented in training, while ACT adjusts wrist orientation more successfully. This suggests that SmolVLA may be overfitting specifically to the wrist orientations seen during training.


![Failure modes by policy](failure_modes_by_policy.png)

## Results of success analysis:
Success modes by policy reveal a striking difference. Comparing ACT and SmolVLA side by side, and separating clean first-attempt successes from more "dirty" successes — including first attempt tipping and second/third attempt pickups - ACT shows substantially more second/third attempt successes than SmolVLA. From the videos, SmolVLA also tends to get closer to the object on subsequent attempts, similar to ACT, but ACT's greater engagement with the object may ultimately make it more successful.

![Success modes by policy](success_modes_by_policy.png)

One question is whether SmolVLA's lower attempt rate is partly caused by synchronous inference. SmolVLA actually replans more frequently than ACT (~2.7 s vs. ~3.3 s per cycle), but spends ~1 s of each cycle waiting for inference rather than executing actions. ACT, in contrast, spends almost the entire episode executing its 100-step chunks. Within a fixed 30 s episode, SmolVLA therefore has substantially less time in active motion, which could reduce the number of opportunities it gets to interact with the object.

![Attempts per episode](attempts_per_episode.png)

To test this, I analyzed the raw joint trajectories, counting visits to the start region and grasp attempts within each visit (tracking the shoulder_pan position and gripper opening). ACT indeed makes more attempts under nominal conditions (2.20 vs. 1.25 per episode). However, SmolVLA converts each attempt at a higher rate: 7 successes / (1.25 × 20 attempts) = 0.28, compared with 9 / (2.20 × 20) = 0.20 for ACT. This suggests that ACT's advantage comes partly from higher interaction throughput, rather than better success per attempt. It would be useful to do async-inference on SmolVLA to test whether removing SmolVLA's inference idle time is enough to close that gap.


## Pre data collection hypotheses vs outcomes

All hypotheses were written down before any grid trials ran (see perturbation_protocol.md).

### Position shift
Prediction was that both policies would struggle in these new regions, as they are far outside the training distribution, with ACT potentially generalizing worse than SmolVLA. However, both policies recognize that the object is outside the trained region and attempt to adjust toward it. During pickup, they tend to go to positions seen during training, often attempting to grasp object in the starting region adjacent to the object's actual location, as if their vision identifies the object's rough location but their "body" cannot follow because that particular action is unfamiliar. This behavior is pronounced for SmolVLA. ACT is somewhat different because it occasionally collides with the object, accidentally pushing or dragging it closer to the trained starting region, where it is more likely to be picked up successfully.

### Novel color
Prediction was that SmolVLA would be more affected by the new color than ACT. This ended up being true: ACT performance was unaffected (45% → 55%), while SmolVLA degraded significantly (35% → 10%). This suggests that ACT may rely more on the object's shape (which is also confirmed by novel object measurement, see below), while SmolVLA, which also takes the task string as input, might be overindexing on color more than I would have hoped ("pink block" is explicitly included in the task description). This SmolVLA hypothesis could be tested by repeating the novel color experiment with SmolVLA, but changing the task description to match the color of the new object.

### Distractor
Hypothesis was that ACT would ignore the sock and SmolVLA would be more distracted, because the sock is pink and color is important conditioning for SmolVLA. It is true that SmolVLA was more distracted. However, it is not true that ACT was unaffected — oscillation mode registered for ACT (3 failures at locations where target object was close to distractor) says it is also registering the distractor. 

### Lighting
Hypothesis was that both policies would degrade, but SmolVLA might be more robust due to its pre-trained prior having seen a wider range of lighting conditions. This predition turned out to be true (ACT 45%→25%, SmolVLA 35%→25%) with SmolVLA also showing many failures that were near_miss.

### Novel object
Hypothesis was that SmolVLA would adjust to a new object better than ACT, given its pre-trained prior. This is exactly what was observed. ACT's most common failure mode was not registering the new object at all (no_target_registered, 16/19 failures), whereas SmolVLA's common failures were near_miss, beside_object, and grasp_slip — all saying it registers and approaches the object, failing either at precise localization (near_miss, beside_object) or at the grasp itself (grasp_slip).

### Cube slide
Hypothesis was that ACT would operate more on "autopilot," going toward the middle of the demo region, while SmolVLA would be better at adjusting to the new position given its expected better "understanding" of the visual world. This is what we observed. Note that success rates in this column are inflated for both policies, as slides moved the block toward more middle of the starting region territory (mostly seen in training).



