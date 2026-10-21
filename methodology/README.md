# Coverage-IW methodology

> Status: research specification; the combined method is not yet an experimentally
> validated contribution. The name **Coverage-IW** is a working name.

## 1. Research question

This study asks:

> Does inverse label-distribution weighting combined with class-coverage-aware
> classifier protection improve rare-class performance under non-IID federated
> learning, while remaining practical on edge hardware?

The task is eight-class blood-cell image classification with BloodMNIST. It is
not a disease-diagnosis study. The primary full-data experiment uses the complete
official training split (11,959 images); the 5K setting is a controlled reduced-
data experiment.

The proposed method addresses two related but different effects of label skew:

1. **Observed-class imbalance:** a client's minibatches can over-represent some
   labels relative to the federation.
2. **Poor class coverage:** a client can have very few or no positive examples
   for a class, but ordinary cross-entropy can still move that class's classifier
   row through negative gradients.

Coverage-IW combines an inverse importance-weighted classification loss for the
first problem with a class-dependent global-head anchor for the second problem.
The server still uses ordinary sample-weighted FedAvg.

## 2. Scope and claims

Coverage-IW is motivated by our independent FedSLD reconstruction and the
existing coverage-calibrated implementation. These distinctions must remain
explicit in code, result tables, and the paper:

- **FedSLD-literal** is our reconstruction of the equation printed in the
  FedSLD paper: local-to-global weights with a batch-size mean.
- **FedSLD-IW** reverses the ratio and normalizes by the sum of weights. This is
  a mathematically motivated reconstruction hypothesis, not a claim about the
  authors' unpublished implementation and not, by itself, a novel method.
- **Coverage** is the project's existing local logit-adjustment plus
  coverage-weighted classifier-head penalty.
- **Coverage-IW** is the proposed combination: normalized inverse importance
  weighting plus coverage-weighted head protection.

The primary Coverage-IW configuration does **not** simultaneously apply the old
local logit adjustment. Applying two label-prior corrections would confound the
mechanism. A combined logit-adjustment variant may be studied only as a clearly
labelled secondary ablation.

## 3. Data protocol

The official dataset split is preserved:

```text
BloodMNIST
├── train ── Dirichlet partition ── client_train[0..K-1]
├── val   ── disjoint monitoring/calibration subsets
└── test  ── untouched until a locked final run
```

- Only the official **training** split is partitioned non-IID across clients.
- A simulated client receives indices for its partition; Flower does not send
  raw training images during a round.
- `client_train` is used for optimization.
- `client_val` is deterministic and used only for monitoring/local early
  stopping where configured.
- Validation selects hyperparameters and checkpoints.
- Test data is never used for training, aggregation, tuning, early stopping, or
  method selection.
- A locked final run evaluates the final selected global model on test exactly
  once. Each seed has its own locked development specification.
- Training augmentation is mild and train-only. Validation and test transforms
  are deterministic. Any fitted normalization statistics use training data only.

In simulation, client partitions represent distinct virtual silos, not twelve
or ten physical hospitals.

## 4. Notation

Let:

- `K` be the number of clients and `C` the number of classes;
- `D_k` be client `k`'s local training set, with size `n_k`;
- `n_kc` be the number of class-`c` training samples on client `k`;
- `N = sum_k n_k` and `n_c = sum_k n_kc`;
- `theta^t` be the global model at round `t`;
- `W_c` and `b_c` be classifier-head parameters for class `c`;
- `B` be a local minibatch and `b_c` its class counts.

All class counts in the loss come from `client_train`. Validation and test
labels never contribute to these quantities.

## 5. Federation prior

Using smoothing `epsilon > 0`, the global training-label prior is

```text
q_c = (n_c + epsilon) / (N + C * epsilon).
```

For simulation, `n_c` is computed from the already assigned training
partitions. For a real deployment, clients would transmit count vectors through
secure aggregation so the server learns only their sum. Class counts are still
sensitive metadata and must not be described as providing formal privacy. No
raw image is required to compute or communicate the aggregate prior.

The global prior is fixed for a fixed training partition and is sent with the
global model/configuration at the start of a round.

## 6. Inverse importance-weighted classification

For each minibatch, define a smoothed batch prior

```text
p_B,c = (b_c + epsilon) / (|B| + C * epsilon).
```

For sample `i` with label `y_i`, the inverse importance weight is

```text
r_i = q_(y_i) / p_B,(y_i).
```

For numerical stability, the implementation may clip this ratio to
`[r_min, r_max]`. Clipping bounds must be fixed using validation and recorded in
the run specification. The normalized classification objective is

```text
L_IW(B) = sum_i r_i * CE(f_theta(x_i), y_i) / sum_i r_i.
```

Normalizing by `sum_i r_i` prevents arbitrary loss-scale changes across
minibatches. A client-prior alternative,
`q_(y_i) / p_k,(y_i)` with `p_k,c` derived from `n_kc`, is a useful stability
ablation, but it is not silently interchangeable with the minibatch version.

FedSLD-literal instead uses the printed local-to-global direction
`p_B,(y_i) / q_(y_i)` and its reported reduction. It remains a separate
comparator.

## 7. Coverage-aware classifier protection

At the beginning of local training in round `t`, client `k` snapshots the
classifier head of the received global model:

```text
W_global^t, b_global^t.
```

For each class, define a coverage-dependent anchor strength

```text
a_kc = kappa / (n_kc + kappa),    kappa > 0.
```

Thus `a_kc = 1` for a locally absent class and decreases as the client obtains
more positive training examples. The head penalty is

```text
L_head,k = (mu / 2) * sum_c a_kc *
           (||W_kc - W_global,c^t||_2^2 +
            ||b_kc - b_global,c^t||_2^2).
```

This term anchors only the final classifier head. The encoder remains free to
learn local features. It cannot create positive evidence for an absent class;
it only limits unsupported classifier drift.

## 8. Proposed local objective

The primary Coverage-IW objective on client `k` is

```text
L_Coverage-IW,k = L_IW + L_head,k.
```

Important boundary cases provide implementation checks:

- `mu = 0` gives FedSLD-IW.
- Uniform `a_kc = 1` gives an IW plus uniform-head proximal control.
- Replacing `L_IW` with ordinary CE gives coverage-head-only.
- Replacing inverse weights with the printed ratio gives the literal
  reconstruction plus head protection, which must be labelled separately.

The training prediction metrics use raw model logits. Weighting and penalties
alter optimization only; they are not applied to validation/test inference.

## 9. One federated round

For round `t = 0, ..., T-1`:

1. The server selects clients and sends `theta^t`, the global training prior,
   and the locked training configuration.
2. Each selected client loads `theta^t` and snapshots its global classifier
   head.
3. The client trains only on `client_train` using Coverage-IW.
4. The client may evaluate on `client_val` for monitoring.
5. The client returns updated model parameters, `n_k`, and scalar metrics.
6. The server aggregates model parameters using sample-weighted FedAvg:

   ```text
   theta^(t+1) = sum_k (n_k / sum_j n_j) * theta_k^(t+1).
   ```

Metrics are aggregated for reporting only. They never replace parameters or
determine the FedAvg weights. After the final round, development runs report
validation results; only a locked final run loads and evaluates the official
test split.

## 10. Experimental comparison

All methods in a comparison block must share the same dataset indices, client
partitions, model initialization, selected clients, rounds, local epochs,
batch size, optimizer, learning-rate budget, augmentation, and seed.

### Main arms

| Arm | Classification loss | Classifier protection | Purpose |
|---|---|---|---|
| FedAvg | CE | None | Standard baseline |
| Balanced sampling | CE with local resampling | None | Strong simple imbalance baseline |
| FedSLD-literal | Printed-ratio weighted CE | None | Transparent paper reconstruction |
| FedSLD-IW | Normalized inverse-weighted CE | None | Importance-weighting baseline |
| Coverage | Existing local logit adjustment | Coverage head anchor | Existing project method |
| **Coverage-IW** | Normalized inverse-weighted CE | Coverage head anchor | Proposed method |

FedProx may be included as an optimization-heterogeneity baseline but does not
replace the label-skew controls above.

### Required ablations

1. FedSLD-IW without head protection (`mu = 0`).
2. Coverage head protection with ordinary CE.
3. IW with a uniform head penalty.
4. Full Coverage-IW.
5. Minibatch prior versus client-level prior.
6. At least one fixed clipping range versus no clipping.

These ablations determine whether any improvement comes from inverse weighting,
generic regularization, coverage targeting, or their interaction.

## 11. Experiment schedule

### Development pilot

- Dataset: full BloodMNIST training split.
- Clients: 10 simulated clients.
- Partition: Dirichlet `alpha = 0.1`.
- Seed: 42.
- Model: GroupNorm TinyCNN at native 64 x 64 resolution.
- Budget: 15-20 rounds, one local epoch per round.
- Selection: validation only; do not load test.

The pilot checks correctness, stability, finite losses/gradients, and whether
the method is promising enough for the full matrix. Pilot results are not the
final evidence.

### Confirmatory matrix

- Dirichlet `alpha` in `{0.1, 0.3}`.
- Seeds `{42, 43, 44}`.
- 10 clients, identical participation schedule per matched run.
- 30 rounds and one local epoch unless the pilot locks another budget for all
  arms before final evaluation.
- Repeat the most severe-skew comparison on the 5K training subset.

Hyperparameters are frozen from validation. Each locked configuration is then
retrained and evaluated once on the official test set. Results must be reported
even when Coverage-IW does not outperform a baseline.

### Architecture and edge study

First establish the algorithmic comparison using the same TinyCNN-GN. Then,
without changing the selected FL hyperparameters opportunistically, compare:

- TinyCNN-GN at 64 x 64;
- MobileNetV3-Small at 64 x 64.

On Jetson Orin and Nano, report batch-one inference latency, local-training
time, peak memory, model/upload bytes, software versions, power mode, and clock
settings. A two-device integration run demonstrates system operation, not
ten-hospital generalization.

## 12. Outcomes and reporting

The primary outcome is **macro-F1**, because accuracy can hide failure on
poorly represented classes. Report:

- macro-F1 (primary);
- worst-class recall and recall/F1 for every class;
- accuracy and balanced accuracy;
- negative log-likelihood, Brier score, and expected calibration error;
- convergence curves and per-round loss stability;
- training time, peak memory, parameter/upload bytes, and total communication;
- mean and sample standard deviation across the three fixed seeds;
- paired per-seed differences between methods sharing the same partition.

Do not choose a model using test accuracy or the best test round. If multiple
hypotheses are tested, describe them and avoid claiming statistical significance
from three seeds without an appropriate analysis.

## 13. Acceptance tests

Before running the expensive matrix, automated tests must verify:

1. 28 x 28 and 64 x 64 loaders preserve split/index identity.
2. Augmentation is enabled only for training.
3. Development runs never load the test split.
4. Class counts and priors use training indices only.
5. Smoothed priors are finite and sum to one.
6. Literal and inverse ratios match hand-calculated examples.
7. Weight normalization and clipping produce finite gradients.
8. Missing classes receive the strongest head anchor.
9. Setting `mu = 0` exactly removes the head penalty.
10. Clients return parameters, the number of training samples, and metrics.
11. FedAvg aggregates parameters by training-sample count, not by metrics.
12. A locked final run evaluates test exactly once after all FL rounds.

## 14. Privacy and limitations

- Federated learning keeps raw images local during rounds, but it does not by
  itself provide formal privacy. Model updates and label-count metadata can leak
  information.
- Secure aggregation, differential privacy, encryption, and adversarial
  robustness are outside the current implementation unless separately added and
  evaluated.
- Synthetic Dirichlet partitions do not reproduce real hospital shifts.
- BloodMNIST contains normal-cell categories and cannot establish clinical
  diagnostic validity.
- Coverage protection cannot learn a class that has no positive examples across
  the entire federation.
- Combining two known mechanisms is not automatically a novel contribution.
  The contribution must be supported by mechanism-focused ablations, consistent
  gains across matched seeds, edge-cost measurements, and an accurate literature
  review.
- An independent site-labelled dataset is required for stronger claims about
  cross-hospital generalization.

## 15. Reproducibility and naming rules

Every result artifact should record:

- git commit and branch;
- strategy name and interpretation;
- dataset version, resolution, split hashes, and selected training indices;
- partition seed/hash and per-client class counts;
- initialization seed and client-selection schedule;
- all loss, optimizer, augmentation, and prior parameters;
- model parameter count and communication accounting;
- whether the run is development or locked final;
- the number of test evaluations.

Use these names consistently:

- `fedsld_literal`: printed-equation reconstruction;
- `fedsld_iw`: inverse-weighted reconstruction hypothesis/baseline;
- `coverage`: existing coverage-calibrated method;
- `coverage_iw`: proposed combined method.

Never label `fedsld_iw` as the original FedSLD implementation. Never report the
forensic stratified-split reproduction as the main BloodMNIST protocol.

## 16. Related project material

- [Research execution workflow](../RESEARCH_STUDY.md)
- [Existing coverage objective](../algorithms/coverage.py)
- [FL data-flow regression tests](../tests/test_fl_data_flow.py)
- [Independent FedSLD reconstruction](../reproductions/FedSLD_2110_08378/README.md)
- [FedSLD paper (arXiv:2110.08378)](https://arxiv.org/abs/2110.08378)
- [FedAvg paper](https://proceedings.mlr.press/v54/mcmahan17a.html)

## One-sentence summary

**Coverage-IW uses a normalized global-to-local label importance ratio to
correct observed-class skew, anchors poorly covered classifier rows to the
round's global head, and retains sample-weighted FedAvg for server aggregation.**
