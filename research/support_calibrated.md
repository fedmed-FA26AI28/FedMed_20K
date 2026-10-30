# Support-calibrated risk benchmark

For the matched 12-client GroupNorm versus BatchNorm experiment, use
[SCR_GN_BN_EXPERIMENT.md](SCR_GN_BN_EXPERIMENT.md).

This directory is an isolated research runner. It does not change the existing
Flower simulation or its saved results. Clients train **sequentially** in one
process, so a 12-client study uses one GPU model at a time. Every development
run uses official MedMNIST `train` for fitting and `val` for checkpoint
selection. The official `test` split is not loaded by `run.py` or `select.py`.
The completed single-seed development runs are summarized in
[PILOT_VALIDATION.md](PILOT_VALIDATION.md).
The default resolution is the datasets' native 28×28.
The default tensor input path batches crop and flip operations on the selected
device. `--slow-data` selects the PIL path, which is needed for the synthetic
client-shift option. If a dataset is missing, add `--download` to
`research.run` or `research.matrix`. New archives are cached under
`data/medmnist/`; existing files in the user's default MedMNIST cache are reused.
All arms use local Adam with a fresh optimizer each round and a common
gradient-norm clip of 5. The printed FedSLD loss retains its batch sum, so
this matched optimizer protocol is distinct from the paper's SGD reproduction.

## Method and claim boundary

For training class counts \(n_{kc}\), let \(n_c=\sum_k n_{kc}\),
\(N=\sum_c n_c\), and \(K\) be the number of clients. For globally supported
classes, the default uniform class target defines

\[
\kappa_c=\gamma n_c/K,\quad
v_{kc}=n_{kc}/(n_{kc}+\kappa_c),\quad
Z_c=\sum_jv_{jc},\quad
\pi_{kc}=v_{kc}/Z_c.
\]

The target risk is
\(R=(1/C)\sum_c\sum_{k:n_{kc}>0}\pi_{kc}\bar\ell_{kc}\).
The client uses fixed sample weight
\(w_{kc}=N/[C Z_c(n_{kc}+\kappa_c)]\) for each locally observed class in
**mean weighted cross-entropy**. It does not renormalize each batch by its
weight sum. The local classifier row is anchored to the received global row
with coefficient \(\lambda(1-v_{kc})\); the anchor penalty is divided by
\(2C\). The encoder remains unanchored and the server uses sample-weighted
FedAvg. Validation and inference use raw logits.

For one simultaneous gradient step at the same global parameters, the
sample-weighted average of these local *data-loss* gradients equals
\(\nabla R\), since
\(\sum_k(n_k/N)(1/n_k)\sum_{i\in D_k}w_{k,y_i}\ell_i
=(1/C)\sum_c\sum_k(v_{kc}/Z_c)\bar\ell_{kc}\).
If class-conditional sample gradients are independent with common variance
\(\sigma_c^2\), the variance contribution scales as
\(\sigma_c^2\sum_k\pi_{kc}^2/n_{kc}\). By Cauchy-Schwarz this term is at
least \(\sigma_c^2/n_c\), with equality at pooled weighting
\(\pi_{kc}=n_{kc}/n_c\). The finite-\(\gamma\) choice therefore trades
variance for more balanced class-specific client influence. These identities
do **not** establish
convergence for multi-step local Adam, clipping, or the head anchor.
\(\gamma\to0\) approaches equal weighting of clients with a class;
\(\gamma\to\infty\) approaches pooled class-balanced CE. With
\(\gamma=\infty\), the head anchor is uniform, which is a crucial control
given the existing project's coverage-calibrated ablation.

The simulation computes training counts from the complete partition for
auditability. `client_support_parameters` constructs each client's weights
from its own counts and the aggregate \(n_c\) and \(Z_c\). A deployment could
aggregate these two vectors without transmitting each client's class counts,
but this runner does **not** implement
secure aggregation or differential privacy. Its virtual clients do not
represent real hospitals.

## Development and final evaluation

Run commands from `FedMed_20K` with the repository's Python environment:

```powershell
& ..\.venv\Scripts\python.exe -m unittest tests.test_support_calibrated -v
& ..\.venv\Scripts\python.exe -m research.audit_partitions

# Full-data pilot; 12 clients are trained sequentially.
& ..\.venv\Scripts\python.exe -m research.run --dataset bloodmnist --partition dirichlet --method support_calibrated --seed 42 --rounds 30 --epochs 1 --gamma 1 --head-lambda 0.01 --device cuda

# Stable PathMNIST practical FedAvg control in the new matched protocol.
& ..\.venv\Scripts\python.exe -m research.run --dataset pathmnist --partition practical --method fedavg --seed 42 --rounds 30 --epochs 1 --resolution 28 --head-lambda 0 --device cuda

# Preview all planned runs; --execute starts them and skips completed runs.
& ..\.venv\Scripts\python.exe -m research.matrix --device cuda
& ..\.venv\Scripts\python.exe -m research.matrix --device cuda --execute

# After every required method and seed has completed, lock hyperparameters
# using validation only, then evaluate each selected checkpoint on test once.
& ..\.venv\Scripts\python.exe -m research.select --dataset bloodmnist --partition dirichlet --device cuda
& ..\.venv\Scripts\python.exe -m research.finalize --lock-file results\research\lock_bloodmnist_dirichlet.json
& ..\.venv\Scripts\python.exe -m research.analyze --lock-file results\research\lock_bloodmnist_dirichlet.json
```

The unrestricted default matrix contains **3,780 sequential development runs**:
three datasets, two partition types, five seeds, 30 rounds, and all
loss/learning-rate settings. It is a substantial compute commitment. Preview
the matrix and run a focused pilot before starting the complete study.
Its method-specific validation grids also cover FedLC \(\tau\in\{0.1,0.5,1\}\)
and FedVLS distillation weight \(\lambda\in\{0.01,0.1,1\}\).
`research.matrix` prints its run count before execution. To start small:

```powershell
& ..\.venv\Scripts\python.exe -m research.matrix --datasets bloodmnist --partitions dirichlet --methods fedavg support_calibrated pooled_uniform_head --seeds 42 --lrs 0.001 --gammas 0.1 1 10 --head-lambdas 0 0.01 --device cuda --execute
```

Pass the same `--lrs`, `--gammas`, `--head-lambdas`, and `--device` values to
`research.select` when locking a restricted matrix. Selection rejects runs
outside those declared grids and checks that selected methods used identical
training partitions for each seed.

Every run writes its training counts, partition hash, configuration, selected
validation round, class metrics, calibration metrics, time, communication
estimate, and model checkpoint under `results/research/`. A round checkpoint
allows a crashed run to resume. The lock requires the requested seeds for
every declared tuning-grid setting of every selected method. `finalize.py` persists each test evaluation and skips
it on resume. The lock records hashes of each selected development file and
checkpoint; finalization verifies both before testing. `analyze.py` reports
paired seed macro-F1 differences and per-class recall. Do not read test
results to choose another variant.
The communication estimate counts model upload and download bytes for all
clients each round; it excludes setup statistics and transport overhead.

`--client-shift-strength 0.2` applies a fixed RGB gain/offset to each client's
training images. Validation and test examples are assigned to synthetic
clients in proportions derived from training class counts and receive the same
client transform. This is a controlled synthetic feature-shift stress test;
test labels determine only the simulated site allocation after the lock and
do not enter fitting or model selection. It does not establish real cross-site
robustness. The main comparison uses zero shift.

```powershell
& ..\.venv\Scripts\python.exe -m research.matrix --datasets bloodmnist --partitions dirichlet --methods fedavg pooled_uniform_head support_calibrated --seeds 42 43 44 45 46 --lrs 0.001 --gammas 1 --head-lambdas 0.01 --client-shift-strength 0.2 --slow-data --execute
& ..\.venv\Scripts\python.exe -m research.select --dataset bloodmnist --partition dirichlet --methods fedavg pooled_uniform_head support_calibrated --lrs 0.001 --client-shift-strength 0.2 --slow-data --lock-file results\research\lock_bloodmnist_dirichlet_shift02.json
```

## Comparators and limitations

The local runner implements FedAvg, the **printed** FedSLD batch/global ratio
with its summation, an independent FedLC logit-margin reconstruction, a
FedVLS loss adapted from the authors' released client code, the
[Coverage-IW](../../coverage_iw.md) minibatch-ratio **plus coverage-head** baseline
(Laplace smoothing \(\epsilon=1\); `--coverage-kappa 10`, with
`--head-lambda 0` reducing to inverse weighting
alone), pooled class-balanced CE with uniform head penalty,
coverage-head-only, support weights plus a uniform head penalty, and the full
support-calibrated risk. The latter pair holds the data risk fixed while
isolating the coverage-dependent head penalty. The FedVLS arm retains this
benchmark's shared backbone, optimizer, and protocol, so it is not an exact
rerun of the authors' framework. Cross-check the FedLC and FedVLS arms
before publication. See the [FedSLD paper](https://arxiv.org/pdf/2110.08378),
[FedLC paper](https://arxiv.org/abs/2209.00189), and
[FedVLS paper](https://arxiv.org/abs/2401.02329) and
[client code](https://github.com/krumpguo/FedVLS/blob/master/system/flcore/clients/clientvls.py).

The `fedswitch_oracle` comparator uses the paper's aggregation score
\(\phi_k=\min(1,\sum_c n_{kc}/\max(1,n_c))\), normalized over clients,
after a first FedAvg round. It uses the **exact** global histogram; the private
histogram estimation and client-level DP of FedSwitch are not implemented.
This arm isolates its aggregation mechanism for a matched comparison and
must be labeled “FedSwitch aggregation oracle,” not FedSwitch itself.
FedProIn is **not implemented**. A faithful feature-shift comparison requires
adapting and validating its official prototype code under the same partitions
and backbone. See [FedSwitch](https://papers.miccai.org/miccai-2026/0383-Paper6676.html),
[FedProIn](https://papers.miccai.org/miccai-2026/0382-Paper1947.html), and
the [FedProIn code](https://github.com/harsh-kmr/FedProIn).

The original FedSLD PathMNIST paper comparison used 80 rounds, five local
epochs, a different CNN and best **test** accuracy. The 30-round, one-epoch
validation-selected research benchmark must not be compared numerically to
that paper table as a direct reproduction. The previous incomplete PathMNIST
simulation runs remain diagnostic only.
