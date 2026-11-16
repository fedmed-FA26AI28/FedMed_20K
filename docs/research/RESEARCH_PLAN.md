# SCR research plan: scarce data and selective drift correction

Updated: 2026-11-13. Status: hypothesis and experiment design; no new candidate
has been validated. Start a new chat with [CHAT_HANDOFF.md](../../CHAT_HANDOFF.md).
Evidence and execution status are tracked in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## 1. Problem and target

An institution may have many images of class A, a few of B, and none of C.
Local training can favor A and damage knowledge of B/C received from the global
model. We want to reduce this damage without suppressing useful local learning
or adding substantial computation and communication.

Distinguish **local scarcity** (other clients have the missing evidence) from
**federation-wide scarcity** (the evidence is insufficient everywhere). Drift
correction can help retain shared knowledge; it cannot create missing positive
examples. Compare with the same architecture trained centrally to diagnose the
FL-specific gap. A centralized result is a reference, not a guaranteed upper bound.

The capstone goal is a defensible improvement under a specified protocol. Beating
one baseline or ranking second on one dataset is not a general SOTA claim.

## 2. What existing SCR does and what remains uncertain

Implementation: [algorithms/scr.py](../../algorithms/scr.py) and
[client/client.py](../../client/client.py).

SCR uses fixed training class counts for weighted cross-entropy and a quadratic
penalty anchoring classifier weights and biases to the received global head.
Its class-support score is

\[
r_{k,c}=\frac{n_{k,c}}{n_{k,c}+\gamma N_c/K}.
\]

Here `n[k,c]` is a client's training count, `N[c]` the federation count, and `K`
the client count. The anchor weight is `1 - r[k,c]`.

For gamma 1, local count 10 with federation average 10 gives the same support
score, 0.5, as local count 1,000 with federation average 1,000. Uniformly scaling
all counts leaves support unchanged. This is a limitation of relative support,
but not proof that it caused the observed performance.

| Hypothesis | What would help distinguish it? |
|---|---|
| Relative support overlooks small absolute counts | Compare original and absolute-count protection with the same ordinary loss. |
| SCR weighting amplifies unreliable evidence or changes the preferred class tradeoff | Separate weighting from anchoring; inspect per-class recall and loss/gradient scales. |
| Normalization explains much of the BloodMNIST/OrganAMNIST gap | Compare FedAvg/FedBN and candidate with/without local BN under the same protocol. |
| Local training overfits or drifts too far | Inspect validation trajectories; separately test fewer local epochs or smaller learning rates. |
| Dataset size drives the apparent difference | Run controlled within-dataset training-size experiments. Cross-dataset rankings cannot isolate size. |

SCR's balancing targets equal class influence, whereas sample-weighted accuracy
weights examples according to their frequency. Better balancing need not increase
BTA. Report class-sensitive metrics alongside accuracy.

## 3. Candidate development, one component at a time

### A. First prototype: absolute-count protection

Use ordinary cross-entropy and replace the relative head-anchor coefficient with

\[
q_{k,c}=\frac{\tau}{n_{k,c}+\tau}, \qquad \tau>0.
\]

A concrete initial objective, preserving SCR's head-penalty normalization, is

\[
L_k=L_{CE}+\frac{\lambda h(t)}{2C}\sum_c q_{k,c}
\left(\|W_{k,c}-W_{g,c}\|_2^2+(b_{k,c}-b_{g,c})^2\right).
\]

`W_g, b_g` are fixed at the start of the round. `h(t)=1` initially; a short
warm-up schedule is a separate later ablation, not an assumed improvement.
Select tau and lambda using validation with a small, declared search budget.
Counts represent distinct training images, not epochs or repeated oversampling.
Counts are an evidence proxy; correlated images need not provide independent evidence.

This changes the strength of an existing idea. It is a diagnostic variant, not
automatically a novel method. It needs one normal forward/backward pass plus head
penalty arithmetic. Actual overhead still requires measurement.

### B. Later prototype: selective update correction

If A is promising, test attenuating only the outward component of an update on
poorly supported classifier rows, rather than penalizing every displacement.
Use the current row's displacement from the received global row as a geometric
reference. Define and test the exact rule before coding it.

Outward motion is not proof of harmful forgetting. A global model can be wrong,
especially early in training. Account for momentum and weight decay: correcting
raw gradients alone need not correct the optimizer's actual update. Handle zero
displacement explicitly. Compare this variant against A; do not silently stack them.

Head protection also does not guarantee preserved predictions: the feature
extractor can drift. In a controlled simulation, use official validation data to
measure class-specific behavior before/after local training. A client with no
local evaluation examples of C cannot measure C-forgetting locally.

### C. Independent factors and deferred work

- Test local BN separately; retain complete institution-specific state for its
  evaluation. Distinguish personalized local-model metrics from one global model.
- Investigate epoch count, learning rate, and batch size separately if diagnostics
  justify it. Fewer examples at fixed epochs also means fewer optimizer steps.
- Consider augmentation or shared pretrained initialization if central training
  also struggles. Give every comparator the same data/pretraining advantage.
- Defer teacher passes, prototypes, synthetic data, and privacy protocols until
  the cheap variants establish a benefit. Avoid an uninterpretable combination.

## 4. Ordered execution checklist

### Step 1 — Reproducibility and evaluation readiness (NEXT)

- [ ] Check branch, working tree, source revision, available strategy files, and CLI flags.
- [ ] Audit selected historical artifacts; reconcile actual settings with command logs.
- [ ] Recover partitions or regenerate with the unchanged algorithm; save content hashes.
- [ ] Define validation data, checkpoint selection, metrics, and hyperparameter budget.
- [ ] Ensure unique round/phase identifiers and honest handling of empty evaluations.

Archive observations remain exploratory if partition identity cannot be verified.
Do not manufacture old hashes or assume the same seed proves identical assignments.

### Step 2 — BloodMNIST diagnostic ablations

Use the reference settings in the handoff. Keep partition, initial weights,
optimizer, seeds, execution backend, and evaluation protocol matched.

| Experiment | Data loss | Head protection | Purpose |
|---|---|---|---|
| FedAvg | Ordinary CE | None | Baseline |
| Original SCR weighting only | SCR weighted CE | None (`head_lambda=0`) | Isolate weighting |
| Original SCR protection only | Ordinary CE | Relative-count anchor | Isolate existing protection |
| Original SCR | SCR weighted CE | Relative-count anchor | Preserve the original comparator |
| Candidate A | Ordinary CE | Absolute-count anchor | Test the scarcity hypothesis |
| FedBN reference | Ordinary CE | None; BN stays local | Diagnose normalization |

- [ ] Add explicit ablation switches/configurations without changing original SCR defaults.
- [ ] Verify coefficient-zero behavior, missing-class handling, raw-logit inference,
  and reproducible sample assignments with focused checks.
- [ ] Prepare Kaggle notebook cells for the user; do not launch long jobs automatically.
- [ ] Run a small correctness pilot, then a limited validation-based screening study.
- [ ] Inspect class support, loss terms, head/backbone drift, actual optimizer steps,
  per-class validation recall, and runtime. Diagnostics must have equal cost across methods.

### Step 3 — Confirm or reject a candidate

- [ ] Freeze the chosen configuration before final comparisons.
- [ ] Complete paired runs over seeds 42–46 on all three datasets at alpha 0.3.
- [ ] Include FedBNS and relevant published methods in the final comparison;
  check implementation fidelity and equal tuning budgets.
- [ ] Report validation-selected test accuracy, macro-F1, balanced accuracy,
  per-class recall, seed variability, runtime, and communication volume.
- [ ] Report BTA/BMCTA separately as exploratory; count real rounds, not duplicate records.
- [ ] Measure time to a predeclared validation target on the same hardware; report failures
  to reach the target. Separate startup/download time from training time.

Keep a candidate only if its benefit is reproducible and its cost justified. If
local BN accounts for the gain, report that attribution instead of crediting the
new correction. No improvement is a valid finding and should narrow the next hypothesis.

### Step 4 — Isolate data availability and extend scope

- [ ] Create deterministic nested training subsets within fixed PathMNIST client
  assignments; preserve validation/test data, label accounting, and nonempty clients.
- [ ] Store each subset's identity separately; never overwrite original partitions.
- [ ] Compare methods at each size, reporting image counts and optimizer-step budgets.
- [ ] Extend a supported result to Dirichlet 0.1 and practical splits as separate studies.
- [ ] Evaluate directional correction B only after a clear baseline for A exists.
- [ ] Treat formal privacy/security as a separate implementation and evaluation gate.

## 5. Relevant references and limits on claims

- [FedSOL, CVPR 2024](https://arxiv.org/abs/2308.12532): preserving global knowledge
  while limiting interference with local learning; [author code](https://github.com/Lee-Gihun/FedSOL).
  A cheap geometric head correction is not equivalent to FedSOL's method.
- [FedSwitch, MICCAI 2026](https://papers.miccai.org/miccai-2026/0383-Paper6676.html):
  private histogram estimation and staged aggregation;
  [author code](https://github.com/Oxcart99-edu/MICCAI_FedSwitch). Its privacy
  guarantees cannot be transferred to SCR by borrowing a switching schedule.
- [FedRS, KDD 2021](https://www.lamda.nju.edu.cn/lixc/papers/FedRS-KDD2021-Lixc.pdf)
  and [reweighted softmax, 2023](https://proceedings.mlr.press/v232/legate23a.html):
  missing-class protection is established prior work; include these in a novelty review.
- [FedBN, ICLR 2021](https://openreview.net/pdf?id=6YEQUn0QICG) and
  [SCAFFOLD, ICML 2020](https://proceedings.mlr.press/v119/karimireddy20a.html):
  separate normalization effects from drift correction.
- [Official MedMNIST definitions](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py):
  preserve official splits, labels, and preprocessing when comparing methods.

SCR currently reads class histograms from simulation partition metadata. Keeping
raw images local, keeping the gate local, or exchanging only model updates does
not itself provide differential privacy or secure aggregation.
