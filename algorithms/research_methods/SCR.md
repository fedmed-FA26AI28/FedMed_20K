# Support-calibrated risk (SCR)

For training count `n[k,c]`, global class count `n[c]`, total count `N`, and
`K` clients, SCR sets

```text
kappa[c] = gamma * n[c] / K
v[k,c] = n[k,c] / (n[k,c] + kappa[c])
pi[k,c] = v[k,c] / sum_j v[j,c]
w[k,c] = N * pi[k,c] / (C * n[k,c])   when n[k,c] > 0
```

Each globally observed class receives `1/C` of the target risk. Clients split
that class's risk according to `pi[k,c]`. A client with no positive examples
of a class has zero data-risk share for that class. Its local classifier row is
anchored to the received global classifier row with coefficient
`head_lambda * (1 - v[k,c])`. Weighted cross-entropy is averaged by batch size
and is **not** divided by the sum of batch weights. The server aggregates the
complete model by sample-weighted FedAvg. Evaluation uses raw logits.

The sample-weighted average of *one simultaneous data-loss gradient step* at a
common model equals the gradient of this class-client risk. This is not a
convergence guarantee for multi-epoch local training, Adam, clipping or
BatchNorm. Training-mode BatchNorm couples examples within each minibatch, so
the BN arm is an empirical test of SCR rather than an exact use of the
per-example gradient identity.

The `baseline_bn` arm shares BatchNorm parameters and running statistics in
FedAvg. It is **global BatchNorm**, not FedBN with client-local statistics.
The two team architectures share convolutions and classifier layers;
BatchNorm adds normalization layers. Common parameter initialization is
matched by seed.

## Matched comparison

| Team architecture | FedAvg loss | SCR loss |
| --- | --- | --- |
| `baseline` | control | proposed method |
| `baseline_bn` | normalization control | proposed method with BN |

Add `pooled_uniform_head` in each architecture as the essential mechanism
control. It uses pooled class-balanced cross-entropy and a uniform classifier
anchor. SCR must beat this control consistently before attributing a gain to
its support-dependent class-client risk. Other implemented comparators include
the printed FedSLD equation, FedLC, FedVLS adaptation and Coverage-IW.

Every arm uses the same official MedMNIST train/validation split, partition
seed, 12 clients, augmentation setting, optimizer search budget, rounds,
epochs, batch size and zero weight decay. The complete commands are in the
[SCR run guide](../../research/SCR_BASELINE_EXPERIMENT.md). The 30-round,
one-epoch matched benchmark is separate from the FedSLD paper's 80-round,
five-epoch reproduction.
