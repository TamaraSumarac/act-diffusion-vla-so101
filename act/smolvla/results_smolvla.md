## Note: camera slot mapping (single-camera fine-tune of a multi-camera pretrained model)

`lerobot/smolvla_base` was pretrained on community robot datasets with three camera
views, expected under the feature names `observation.images.camera1/2/3`. Our rig has
a single camera (`observation.images.front`). At launch, the dataset's camera was
mapped onto the first expected slot via
`--rename_map='{"observation.images.front": "observation.images.camera1"}'`;
the remaining views are handled as absent, per SmolVLA's variable-camera design
(the architecture was built for heterogeneous community data where camera counts
differ across datasets).

Two implications worth keeping in view when reading SmolVLA's results:
1. **This is the first structural asymmetry of the three-way comparison.** ACT and
   Diffusion Policy were constructed *around* this dataset's features; SmolVLA
   arrives with expectations inherited from pretraining that the dataset must be
   adapted into. That is not a bug — it is the cost side of the pretrained-prior
   trade this comparison exists to measure.
2. **The pretrained visual prior is being consumed under narrower conditions than
   it was formed.** The model's pretraining saw multi-view scenes; our fine-tune
   shows it one view in slot 1 with the others empty. If SmolVLA underperforms,
   "prior formed on three views, deployed on one" is a candidate explanation to
   weigh alongside the usual ones — and if it outperforms ACT's task-trained
   vision anyway, the result is stronger for it.