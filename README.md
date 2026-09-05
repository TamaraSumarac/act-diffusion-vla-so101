# ACT vs SmolVLA: How imitation learning policies generalize beyond their demonstrations

Over the past couple of years I have been impressed by what imitation learning can get robots to do from demonstrations alone - folding laundry, tying trash bags, and more. This raised a question **how do different imitation learning approaches, trained on identical demonstrations, compare in their learned behavior, generalization, and failure modes?**

To explore that question, I trained three policies on an SO-101 arm: ACT and Diffusion, both trained from scratch, and SmolVLA, a pre-trained model fine-tuned for this experiment. All three were trained on **the same frozen dataset of only 50 pick and place demonstrations**. ACT and SmolVLA were then evaluated on real hardware across a grid of six perturbations (272 scored trials). Diffusion was evaluated under nominal conditions only, as deployment constraints prevented evaluation across the perturbation grid (see Diffusion section below). Goal of this work was not just to compare success rates, but to understand **what each policy learned from the same limited data, how well that behavior generalized, and what their specific failures reveal about their strengths and weaknesses.**


![Robustness heatmap](perturbation/robustness_heatmap.png)

**TLDR:** 
- Two policies seem to learn different features from the same data. ACT appears to rely more on object shape than color, while SmolVLA is more sensitive to color (partially through the task string, but also through its visual model; see follow-up results below). In terms of trajectory, ACT seems to operate more on “autopilot,” often reproducing an “average” of the training trajectories, whereas SmolVLA appears more reactive to its visual observations. Both policies show that positional control could be significantly improved with better training data coverage (supported by the SmolVLA follow-up results below). 
- At first glance, the attempt statistics suggest a quantity vs quality tradeoff: ACT makes more grasp attempts, while SmolVLA converts each attempt better. Follow-ups below complicate that framing. ACT’s extra attempts are largely a byproduct of replaying an averaged trajectory focusing on middle of the the object start region — persistent, but not aimed. SmolVLA’s attempts are aimed but imprecise, with precision rather than opportunity being the main limitation: increasing its attempt rate via async inference did not improve success, while better training-data coverage did. 
- Diffusion Policy, evaluated under nominal conditions only, qualitatively showed the most within episode adaptation — visibly correcting its approach rather than replaying an averaged trajectory. However, its deployment never reached an operating point where this could be scored under perturbation.



![Novel object, side by side](perturbation/novel_object_side_by_side.gif)

*Video of novel object pickup with ACT (left) and SmolVLA (right). This example shows SmolVLA generalizing better to a novel object, suggesting a stronger visual representation of the environment. Both policies see the same condition - a pink sock representing the target object replacing the pink block that both policies saw in training - but behave very differently. ACT hovers between the starting region and target tray, as if it registers that target object is absent from both locations, suggesting it never recognizes the sock as a target object. SmolVLA, in contrast, localizes the sock and attempts a grasp, nearly succeeding.*

## Key takeaways
- **Nominal success rates hide the differences.** Under nominal conditions, ACT and SmolVLA have similar success rates (45% vs. 35%), while Diffusion is lagging behind (10% pick and place, 25% pick-up) — though its numbers reflect a constrained deployment more than the policy itself (see Diffusion section). But *how* these policies fail already reveals differences between them, and these become much clearer under conditions none of them saw in training.
- **Training data is the most important success factor.** As expected, and consistent with where much of the field is focused, training data that covers the range of possible conditions strongly affects success. Shifting the target object just a few centimeters outside the training start region drops both ACT and SmolVLA policies to ~0% (1/16 and 0/16). ACT gets more accidental collisions with the target; SmolVLA partially localizes it but cannot construct a grasp out of distribution. In follow-up, 16 additional demos covering the edges of the start region and more wrist angles (none of the tested nominal positions, to avoid contaminating the eval) was the single intervention that significantly improved SmolVLA — 35% → 55% from a 32% increase in data.
- **Color vs. shape.** Policies rely on different visual features. A different color block drops SmolVLA from 35%→10% while leaving ACT unchanged; a different shape (pink block → pink sock) does the opposite — ACT fails to register the sock as a target in 16/19 trials, while SmolVLA still localizes and reaches it. A follow-up probe with the instruction matching the actual color recovers part but not all of SmolVLA's drop, so the color sensitivity enters partly through the task string and partly through the visual model.
- **Representation beats reaction time.** In the mid-reach slide condition, ACT inference is ~3× faster (~351 ms vs. ~1 s), giving it more opportunities to replan. Despite this, ACT does not adjust its trajectory to re-target the moved object, while SmolVLA does.
- **Quantity vs quality of attemps.** From joint traces, ACT makes ~1.8× more grasp attempts per episode, while SmolVLA converts each attempt better (0.28 vs. 0.20). This first read like a quantity vs quality tradeoff, but the follow-ups dissolve it: ACT's extra attempts are largely accidental sweeps of an averaged trajectory, and SmolVLA's limit is precision, not opportunity — running it asynchronously raised its attempt rate but not its success, while better data coverage raised success.
<!-- - **Deployment can dominate evaluation.** Diffusion's low nominal score says less about the policy than about its synchronous deployment: action-chunk seams produced joint jumps of up to ~38°, largely independent of the denoising scheduler, and the safety clamp this forced interacted with the episode budget to cap success. Its most striking behavior — visibly correcting its approach within an episode rather than replaying an averaged trajectory — is exactly what the perturbation grid is designed to measure, once deployment reaches parity (see Diffusion section). -->

## Method

- **Hardware:** SO-101 leader/follower arm pair, single front facing camera (positioned across from the follower), LeRobot v0.6.1.
- **Dataset:** 50 teleoperated pick and place demos, frozen before any training ([`so101_policy_robustness`](https://huggingface.co/datasets/TamaraSumarac/so101_policy_robustness)). All perturbation comparison results use this dataset only. For the coverage follow-up, 16 additional demos exploring the edges of the start region and a wider range of wrist angles (deliberately excluding the nominal positions used in the eval) were collected as [`so101_policy_robustness_v2`](https://huggingface.co/datasets/TamaraSumarac/so101_policy_robustness_v2) and used to retrain SmolVLA.
- **Policies:** LeRobot default configurations throughout. ACT and Diffusion Policy trained from scratch; SmolVLA fine-tuned from its pre-trained checkpoint. Diffusion deployed with DDIM-50 and a `max_relative_target=8` safety clamp (see Diffusion section).
- **Eval:** perturbation protocol details ([protocol](perturbation/perturbation_protocol.md)) — nominal conditions for all three policies (20 trials each); the perturbation grid of 6 new conditions (position shift, novel color, pink sock distractor, lighting, novel object, mid-reach slide) × 20 trials (16 for position shift) for ACT and SmolVLA. Binary scoring, fixed trial order, no discards. Failure/success taxonomy coded per trial; attempt counts measured from joint traces. Follow-up experiments (task-string probe, async inference, v2 retrain) described in their section below.

## Results:

Full analysis of ACT and SmolVLA comparison in [perturbation/perturbation_results.md](perturbation/perturbation_results.md): per condition results, failure mode and success mode analysis, joint trace based attempt/conversion analysis, experiment hypothesis discussion. Raw trial log and analysis notebook in [`perturbation/`](perturbation/).

## Follow-up work
- **Task-string probe for SmolVLA:** Matching the task string to the novel color doubled success (10% → 20%) and eliminated the "block not registered" failure mode — remaining failures were SmolVLA's usual precision errors. This didn't fully recover the nominal result (35%), so color sensitivity enters partly through the instruction and partly through the visual model. Full results in [smolvla_novel_color_task_string.md](perturbation/follow-up%20results/smolvla_novel_color_task_string.md).
- **Async inference for SmolVLA:** Grid results suggested that if SmolVLA could be latency limited and that raising number of pick-up attempts could raise success substantially (~60% at its 0.28/attempt conversion, if attempts reached ACT's rate). Async inference raised attempts by 28%, but success didn't move (30% vs. 35% nominal) — refuting the quantity of attemps hypothesis. The traces suggest why: ACT's extra attempts often shove the block into easier pickup positions by accident, while SmolVLA's extra attempts re-adjust to the object often without displacing it — more deliberate, but still limited by positioning precision. Full results in [smolvla_remote_inference.md](perturbation/follow-up%20results/smolvla_remote_inference.md).
- **More and better data coverage improves SmolVLA:** 16 additional demos covering the edges of the start region and wider wrist angles (none at the tested positions or angles) raised nominal success from 35% to 55%, with the clearest gain in wrist-angle handling (e.g. the 135° grasp the v1 model never managed). Full results in [smolvla_more_demo_data.md](perturbation/follow-up%20results/smolvla_more_demo_data.md).

## Diffusion:
- XX

## Connection to earlier work
This is a hardware follow-up to my [domain-randomization ablation study](https://github.com/TamaraSumarac/dr-ablation-study) on a simulated Unitree Go2. Both studies explore a similar question: **nominal performance alone does not tell us much about what a policy learned — the differences become much clearer when we test policies in conditions they never explored in training.** In simulation, policies trained with and without individual domain randomizations performed similarly under nominal conditions, but differences emerged once the physics was perturbed. Here, ACT and SmolVLA are trained on identical data and again show relatively similar nominal performance, while perturbing the scene reveals very different learned behaviors.

There is another similarity between the two studies: looking at only one perturbation at a time would miss some of the more interesting results. In simulation, this revealed a coupling between friction and motor strength; here, testing both color and shape changes reveals that the two policies seem to rely on different visual features. Similarly, the most obvious explanation is not always the right one: more randomization was not necessarily better in simulation, and faster inference does not necessarily lead to better re-targeting here. In both cases, it seems like **understanding what the policy learned matters more than nominal performance alone.**


## Next steps

1. **Task-string probe for SmolVLA:** rerun novel color with the instruction matching the actual color - if success rate similar to nominal this confirms SmolVLA indexes on task string significantly.
2. **Async inference for SmolVLA:** In order to test if quantity beats quality here (hypothesis of this work) the goal of this test would be to raise SmolVLA's attempt throughput; at current conversion (0.28/attempt) nominal success should rise substantially (~60% if attempts reach ACT's rate).
3. **Wrist-orientation overfitting for SmolVLA:** SmolVLA seems to go for the 0°-style grasps across conditions suggesting overfitting; candidate fixes are wider orientation coverage in data (so ~20 additional demos at 45/90/135°) or orientation-consistent augmentation (rotating the image and the wrist label together to synthesize demos at unseen orientations).
4. **Diffusion Policy deployment:** Diffusion was excluded from this comparison because I ran into deployment issues with synchronous inference. Discontinuities between action chunks were so large, that it made arm motion jerky enough to physically damage the arm (the gripper motor detached and the wrist servo connection was damaged), so I stopped further evaluation. My next step is to reduce the observation to action staleness that is likely causing these discontinuities, using LeRobot’s async inference and a faster GPU (rented). If that resolves the issue, I will evaluate Diffusion on the same frozen perturbation grid and complete the three-policy comparison.


## Repo map

| Path | What |
|---|---|
| `perturbation/perturbation_protocol.md` | Frozen pre-registered eval protocol |
| `perturbation/perturbation_results.md` | Full results + hypothesis verdicts |
| `perturbation/trial_log.xlsx` | Per-trial log: scores, failure notes, episode indices |
| `perturbation/DataAnalysis.ipynb` | Heatmaps, taxonomy coding, attempt analysis |
| `perturbation/videos/` | Result clips pulled from eval episodes |
| `act/`, `smolvla/`, `diffusion/` | Per-policy training/eval configs and notes |
| `tools/` | Eval launchers, calibration gate, rollout scripts |
| `rig/` | Hardware setup and collection protocol |

Eval rollout datasets (all 272 episodes, video + joint traces): [HF Hub](https://huggingface.co/TamaraSumarac) under `rollout_*`.