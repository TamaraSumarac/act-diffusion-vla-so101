# Postmortem: Diffusion Policy NaN — a hardware fault wearing a config costume

**Date:** 2026-08-20 · **Context:** launching Diffusion Policy training (LeRobot 0.6.1, stock
preset, 263M params) on the same frozen dataset, trim pipeline, and Lambda A10 recipe that had
trained ACT flawlessly three days earlier. · **Symptom:** `loss:nan grdn:nan` by step 200, every
run. · **Resolution:** the physical machine was faulty. Every config theory was wrong.

---

## Symptom

First Diffusion launch died with NaN loss and NaN gradient norm at the first log line (step 200),
during LR warmup (lr ≈ 2e-5, far below the 1e-4 peak). ACT had trained 100k clean steps on the
same box, same venv, same dataset, same sampler-level trim — so the initial suspect list was
everything Diffusion-specific.

## What we tried, in order (and what each attempt taught)

### 1. Batch size (theory: small-batch gradient variance)
Stock preset is calibrated for large batches; at batch 8, per-step gradient variance is high, and
the preset has no gradient clipping (`grad_clip_norm` absent from `DiffusionConfig`).
**Test:** relaunch at `--batch_size=32`. **Result:** NaN at step 200 anyway, still in warmup.
**Verdict:** batch size innocent. A 4× change producing zero difference killed the variance theory.

### 2. Isolate the forward pass (theory: bad data or broken op)
Minimal probe: construct the policy manually, load one real batch, one forward pass, print
input finiteness and loss.

```python
# (abridged) — one forward pass on one real batch
policy = make_policy(DiffusionConfig(device="cuda"), ds_meta=meta)
batch = next(iter(DataLoader(ds, batch_size=8, shuffle=True)))
for k, v in batch.items():
    if isinstance(v, torch.Tensor) and v.is_floating_point():
        print(k, torch.isfinite(v).all().item(), v.min().item(), v.max().item())
loss, _ = policy.forward(batch)
print("loss:", loss.item())
```

**Result:** all inputs finite, loss = 1.18. Forward pass healthy. The blow-up needed the
training loop.

### 3. Manual training loop probe (theory: optimizer-step blow-up)
Extended the probe to real optimizer steps with the preset's exact settings
(Adam, lr=1e-4, betas=(0.95, 0.999), eps=1e-8), instrumented to check gradients for NaN
**before** the optimizer step and weights **after** it:

```python
for step in range(50):
    loss, _ = policy.forward(batch)
    opt.zero_grad(); loss.backward()
    bad_grads   = [n for n, p in policy.named_parameters()
                   if p.grad is not None and not torch.isfinite(p.grad).all()]
    opt.step()
    bad_weights = [n for n, p in policy.named_parameters()
                   if not torch.isfinite(p).all()]
    print(step, loss.item(), len(bad_grads), len(bad_weights))
```

**Result:** stochastic — one run died at step 4, another ran 6 steps clean, a third died at
step 8 with `bad_grads=148` **before** the optimizer step. Loss at the fatal step was a healthy
1.02. Key facts extracted: (a) gradients go NaN in **backward**, optimizer merely inherits them;
(b) the failure is **stochastic** (batch-order dependent); (c) inputs are finite every time.

### 4. BatchNorm theory (wrong, but educational)
The dead-parameter list started at `rgb_encoder.backbone...bn1`, and the Diffusion Policy paper
itself replaces BatchNorm with GroupNorm in the vision encoder for stability. Convicted BN.
**Test:** `--policy.use_group_norm=true` (which forced `--policy.pretrained_backbone_weights=null`
— GN can't be swapped into BN-shaped pretrained weights; from-scratch+GN is the paper's own
configuration). **Result:** NaN at step 200 anyway. **Verdict:** BN innocent. *Post-hoc
realization:* the "first dead layers" were an ordering artifact — `named_parameters()` starts at
the encoder, so with 148 dead params the true origin could be anywhere downstream.

### 5. Exonerate our own pipeline (control experiment)
Ran **stock** `lerobot-train` directly — no custom launcher, no sampler trim — on the same
dataset. **Result:** NaN at step 200. **Verdict:** our trim/launcher fully exonerated. The bug is
in stock LeRobot + this dataset + this machine.

### 6. Name the guilty op (autograd anomaly detection)
```python
torch.autograd.set_detect_anomaly(True)   # traces backward, raises at the first NaN-producing op
```
**Result:** `RuntimeError: Function 'MseLossBackward0' returned nan values in its 0th output` —
with the printed loss still finite. Interpretation: the elementwise MSE (reduction="none", then
padding-masked) contained NaN/inf at individual elements that the masked reduction *excluded* —
so the displayed loss stayed clean while backward through the masked elements produced inf×0=NaN.
The NaN was born in the U-Net's output at scattered elements, hidden under the padding mask.

### 7. Torch version pin (theory: torch 2.11 + CUDA 13.0 numerics regression)
Community runs diffusion on torch 2.5–2.7/cu12x; our box ran a very new 2.11+cu130 stack.
**Test:** `pip install torch==2.7.1 torchvision==0.22.1 --index-url .../whl/cu126`, rerun stock
probe. **Result:** NaN, identical. **Verdict:** torch version innocent.

### 8. The decisive two-variable split: reference dataset, then reference machine
Only two suspects remained: this dataset and this machine.
**Test A:** stock diffusion on `lerobot/pusht` (the dataset LeRobot's implementation was
validated on) on the same box. **Result: NaN at step 200.** Dataset exonerated — stock training
on the reference dataset fails on this box.
**Test B:** terminate the box, launch a fresh A10, identical software recipe, rerun the pusht
probe. **Result: clean** — loss 0.390 → 0.060 over 1000 steps, healthy gradient norms.
**Then the real launch on the new box, fully stock config: trains cleanly.**

## Root cause

A hardware/driver-level compute fault on one specific Lambda A10 instance, producing stochastic
NaN in Diffusion Policy's op mix (263M-param U-Net) while leaving ACT's op mix (52M transformer)
unaffected. Not reproducible from software; not fixable from inside the box.

## Why it was hard

Every early symptom fit a plausible *software* story: small batch + no grad clip + preset LR
(classic recipe for instability), BN in a diffusion vision encoder (the paper itself warns about
it), a bleeding-edge torch/CUDA stack. Three well-motivated theories in a row were consistent
with the evidence and wrong. The stochasticity made single negative results weak (a clean probe
run proved nothing), and the padding mask actively hid the NaN from the printed loss.

## What actually worked: escalating isolation

1. Forward pass alone → healthy (localizes to training loop)
2. Manual loop with per-step grad/weight finiteness checks → NaN born in backward, stochastic
3. `torch.autograd.set_detect_anomaly(True)` → names the exact op
4. Control: stock trainer without our code → our pipeline exonerated
5. Control: reference dataset → our dataset exonerated
6. Control: reference machine → box convicted

## Lessons

- **Run the reference probe first.** Stock policy + reference dataset (`lerobot/pusht`) is a
  10-minute experiment that cleaves the world into "my stuff" vs "the platform." We ran it
  seventh; it should have been second. It's now step one in the cloud runbook for any
  new-policy-on-new-box situation.
- **Hardware faults mimic config bugs.** "ACT trained fine on this box" felt like it exonerated
  the machine — it only exonerated the machine *for ACT's ops*. Different architectures exercise
  different kernels.
- **Beware ordering artifacts in evidence.** The `named_parameters()` list starting at the vision
  backbone made BN look guilty; with 148 dead tensors, "first in the list" ≠ "first to die."
- **Stochastic bugs need repetition or instrumentation, not single runs.** One clean probe
  after a config change proves nothing; the anomaly detector converts a stochastic failure into
  a deterministic name.
- **Masked losses can hide NaN.** A finite printed loss is not proof of a finite loss tensor when
  reduction="none" + masking is involved.
- When a platform is broken below your access level, **leave** — the fix was `terminate` and a
  fresh instance, the same lesson as the Phase 1 Vulkan/driver saga, one level deeper in the stack.

## Interview-sized version (60 seconds)

"Launching Diffusion Policy on the same pipeline that had just trained ACT cleanly, I hit NaN by
step 200 on every run. I worked through the plausible software causes with controlled probes —
batch size, learning rate, the paper's known BatchNorm instability, even pinning PyTorch back two
versions — and instrumented a manual training loop that showed gradients going NaN in the
backward pass, stochastically, with finite inputs and a finite displayed loss; autograd anomaly
detection traced it to masked elements of the elementwise MSE. The decisive move was a
two-variable isolation: stock training on LeRobot's reference dataset *also* NaN'd on that
machine, and the identical recipe on a fresh instance trained perfectly — the 'bug' was a faulty
GPU instance. The takeaway I operationalized: a reference-dataset probe is now step one of my
cloud runbook, because it splits 'my code' from 'the platform' in ten minutes, and I'd spent an
afternoon learning that the hard way."
