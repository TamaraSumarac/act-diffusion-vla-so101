# ACT vs SmolVLA: How imitation learning policies generalize beyond their demonstrations

Imitation learning has become remarkably effective at teaching robots new manipulation tasks from demonstrations. This raises the question of **how different imitation learning approaches trained on identical demonstrations compare in their learned behavior, generalization, and failure modes.**

To explore that question, I trained two policies on an SO-101 arm: ACT, trained from scratch, and SmolVLA, a pre-trained vision-language-action model that was fine-tuned for this experiment. Both policies are fed **identical frozen dataset of only 50 pick-and-place demonstrations**. I then evaluated them on real hardware (SO-101 arm) across a grid of six perturbations, for a total of 272 scored trials. The goal was not just to compare success rates, but to understand **what each policy had learned from the same limited data, how well that behavior generalized, and what their failures reveal about the two approaches.**


![Robustness heatmap](perturbation/robustness_heatmap.png)

**TLDR:** two policies seem to learn different features from the same data. ACT appears to rely more on object shape than color, while SmolVLA is more sensitive to color—though this may be influenced by the language instruction. In terms of trajectory, ACT seems to operate more on “autopilot,” often reproducing an “average” of the training trajectories, whereas SmolVLA appears more reactive to its visual observations. Both policies show that positional control could be significantly improved with better coverage in the training data. At first glance, SmolVLA seems to have a better visual representation of the environment, while ACT engages the object more persistently - more attempts and more retries, though not always deliberate ones.


![Novel object, side by side](perturbation/novel_object_side_by_side.gif)

*Video of novel object pickup with ACT (left) and SmolVLA (right). This example shows SmolVLA generalizing better to a novel object, suggesting a stronger visual representation of the environment. Both policies see the same condition - a pink sock representing the target object replacing the pink block that both policies saw in training - but behave very differently. ACT hovers between the starting region and target tray, as if it registers that target object is absent from both locations, suggesting it never recognizes the sock as a target object. SmolVLA, in contrast, localizes the sock and attempts a grasp, nearly succeeding.*

## Key takeaways

- **Nominal success rate hide the differences.** Under nominal conditions, the two policies have similar success rates (45% vs. 35%). But how they fail already reveals differences between them, which become much clearer under conditions neither policy saw during training.
- **Training data is the most important success factor.** As expected, and consistent with where much of the field is focused, having high quality training data that covers the range of possible conditions strongly affects success. Shifting the target object just a few centimeters outside the start region used in training drops both policies to ~0% success (1/16 and 0/16). ACT gets more accidental collisions with the target object, while SmolVLA partially localizes it but cannot construct a successful grasp out of distribution.
- **Color vs. shape.** Policies appear to rely on different visual features. Different color block drops SmolVLA from 35%→10% while leaving ACT unchanged. Changing the object shape instead (pink block → pink sock) has the opposite effect: ACT fails to register the sock as a target in 16/19 trials, while SmolVLA still localizes it and reaches it.
- **Representation beats reaction time.** In the mid-reach slide condition, ACT inference is ~3× faster (~351 ms vs. ~1 s), giving it more opportunities to replan. Despite this, ACT does not adjust its trajectory to re-target the moved object, while SmolVLA does.
- **Throughput vs quality:** ACT and SmolVLA differ in the quantity vs quality of their grasp attempts. Based on joint traces, ACT makes ~1.8× more attempts per episode, while SmolVLA converts individual attempts more successfully (0.28 vs. 0.20 success per attempt). This suggests ACT's higher overall success rate may be driven partly by having more attempts, while SmolVLA may be latency-limited rather than less capable per attempt - something that could be tested with asynchronous inference.

## Method

- **Hardware:** SO-101 leader/follower arm pair, single front-facing camera (positioned across from the follower), LeRobot v0.6.1.
- **Dataset:** 50 teleoperated pick-and-place demos, frozen before any training ([`so101_policy_robustness`](https://huggingface.co/datasets/TamaraSumarac/so101_policy_robustness)).
- **Policies:** LeRobot default configurations for ACT and SmolVLA fine-tune, trained on the identical dataset. (Diffusion Policy was excluded at the deployment gate after sync-inference chunk
  disagreement physically damaged a servo; its eval is scoped to follow up work.)
- **Eval:** perturbation protocol details ([protocol](perturbation/perturbation_protocol.md)) — 6 new conditions (position shift, novel color, pink sock distractor, lighting, novel object, mid-reach slide) × 2 policies × 20 trials (16 for position shift), binary scoring, fixed trial order, no discards. Failure/success taxonomy coded per trial; attempt counts measured from joint traces.

## Results

Full analysis in [perturbation_results.md](perturbation/perturbation_results.md): per condition results, failure mode and success mode analysis, joint trace based attempt/conversion analysis, experiment hypothesis discussion. Raw trial log and analysis notebook in [`perturbation/`](perturbation/).

## Connection to earlier work
This is a hardware follow-up to my [domain-randomization ablation study](https://github.com/TamaraSumarac/dr-ablation-study) on a simulated Unitree Go2. Both studies explore a similar question: **nominal performance alone does not tell us much about what a policy learned — the differences become much clearer when we test policies in conditions they never explored in training.** In simulation, policies trained with and without individual domain randomizations performed similarly under nominal conditions, but differences emerged once the physics was perturbed. Here, ACT and SmolVLA are trained on identical data and again show relatively similar nominal performance, while perturbing the scene reveals very different learned behaviors.

There is another similarity between the two studies: looking at only one perturbation at a time would miss some of the more interesting results. In simulation, this revealed a coupling between friction and motor strength; here, testing both color and shape changes reveals that the two policies seem to rely on different visual features. Similarly, the most obvious explanation is not always the right one: more randomization was not necessarily better in simulation, and faster inference does not necessarily lead to better re-targeting here. In both cases, it seems like **understanding what the policy learned matters more than nominal performance alone.**



## Next steps

1. **Task-string probe for SmolVLA:** rerun novel color with the instruction matching the actual color - if success rate similar to nominal this confirms SmolVLA indexes on task string significantly.
2. **Async inference for SmolVLA:** In order to test if quantity beats quality here (hypothesis of this work) the goal of this test would be to raise SmolVLA's attempt throughput; at current conversion (0.28/attempt) nominal success should rise substantially (~60% if attempts reach ACT's rate).
3. **Wrist-orientation overfitting for SmolVLA:** SmolVLA seems to go for the 0°-style grasps across conditions suggesting overfitting;
   candidate fixes are wider orientation coverage in data (so ~20 additional demos at 45/90/135°) or orientation-consistent augmentation (rotating the image and the wrist label together to synthesize demos at unseen orientations).
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