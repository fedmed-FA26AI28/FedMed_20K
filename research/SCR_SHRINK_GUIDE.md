# Guided SCR correction experiment

Use this guide from the `FedMed_20K` repository root. On Kaggle, replace
`python` with the notebook's Python executable if needed. Keep the same 12
clients, official MedMNIST splits, partition seed, model, optimizer, rounds,
epochs, batch size, augmentation setting, and zero weight decay in every arm.
The partition functions are unchanged.

## What is being tested

`support_shrink` is a separate candidate method. For every locally observed
class it uses `weight = (1-rho) + rho * SCR_weight`, with the existing SCR
classifier anchor. `rho=1` recovers the SCR data loss. `rho=0` with
`head_lambda=0` recovers the FedAvg objective. The existing
`support_calibrated` method remains the unmodified SCR control.

The 30-round, one-epoch matched study below is separate from the paper-style
80-round, five-epoch SGD reproduction. Do not compare their scores directly.

## Step 1: preflight

Fetch the official BloodMNIST archive once if it is not already cached. On
Kaggle this requires an Internet-enabled session or an attached copy of the
archive. Downloading does not change the official split.

```bash
python -c "from research.data import get_medmnist_datasets; get_medmnist_datasets('bloodmnist', download=True, load_val=False, load_test=False)"
python -m unittest tests.test_research_method_registry tests.test_support_calibrated -v
python -m research.audit_partitions --datasets bloodmnist --seeds 42 --clients 12 --alpha 0.1
python -m research.matrix --list-methods
```

The partition audit must report 12 nonempty clients with every training row
assigned exactly once. For a given dataset, partition, alpha, and seed, all
methods in the matrix regenerate exactly the same client indices. Every
development result also records `partition_sha256` for the final check.

## Step 2: eight-run BloodMNIST pilot

Preview first. The same command with `--execute` starts sequential training.
This is one seed and one setting, so treat it as a pipeline and mechanism
pilot, not evidence of a method gain.

```bash
python -m research.matrix --datasets bloodmnist --partitions practical --models baseline baseline_bn --methods fedavg pooled_uniform_head support_calibrated support_shrink --seeds 42 --lrs 0.001 --gammas 10 --head-lambdas 0.01 --rhos 0.5 --clients 12 --rounds 30 --epochs 1 --batch-size 64 --optimizer adam --weight-decay 0 --device cuda --output-root results/research_scr_shrink_blood_practical

python -m research.matrix --datasets bloodmnist --partitions practical --models baseline baseline_bn --methods fedavg pooled_uniform_head support_calibrated support_shrink --seeds 42 --lrs 0.001 --gammas 10 --head-lambdas 0.01 --rhos 0.5 --clients 12 --rounds 30 --epochs 1 --batch-size 64 --optimizer adam --weight-decay 0 --device cuda --output-root results/research_scr_shrink_blood_practical --execute
```

Completed runs are skipped when the same command is rerun. Interrupted runs
resume from their saved round. The matrix prints a
`validation_comparison.png` path; each run also has `validation_curves.png`.
The comparison plot includes validation macro-F1 against summed round time,
and `validation_summary.csv` records `training_round_seconds`. Those times
exclude process startup and data loading; use full wall-clock time as a
separate systems measure.
To regenerate plots after copying results, run:

```bash
python -m research.plots --output-root results/research_scr_shrink_blood_practical --dataset bloodmnist --partition practical
```

Read validation macro-F1 first, then accuracy and worst-class recall. Compare
`support_shrink` with FedAvg, original SCR, and the pooled uniform-head
control **within the same model**. Also inspect the per-run
`development.json` `partition_sha256` and `train_class_counts`. A lower
training loss alone does not establish an improvement.

## Speed and accuracy before PathMNIST

The older Flower SCR runs used `experiments.simulation_client`, which locks
the GPU so 12 client processes train and evaluate one at a time. Several older
FedAvg runs used `client.client`, allowing concurrent execution on different
hardware. Their wall-clock ratio does not measure the cost of SCR's loss.
For BloodMNIST seed 42 with `baseline_bn`, the saved Flower round medians are
1.38 seconds for FedAvg and 15.76 seconds for SCR; FedAvg's summed client
epoch time exceeds its wall time because clients overlap, while SCR's epoch
time nearly equals wall time because the GPU lock serializes them. Re-run
both methods in one execution path before reporting a speed ratio.
The research runner uses one process with sequential clients and raw tensor
input by default; leave `--slow-data` off for the matched pilot. It also
avoids copying minibatch loss and validation probabilities back to the CPU
after every batch. Keep these execution settings identical across methods.

Before a long PathMNIST matrix, run the same full-data FedAvg protocol for
three rounds to measure local throughput. Ignore the first round when
estimating steady-state time. Use a separate output root and do not interpret
its three-round score as a final result:

```bash
python -m research.run --dataset pathmnist --partition practical --method fedavg --model baseline_bn --seed 42 --clients 12 --rounds 3 --epochs 1 --batch-size 64 --lr 0.001 --optimizer adam --weight-decay 0 --device cuda --download --output-root results/path_speed_probe
```

If the Path pilot is still too slow, compare batch sizes 64, 128 and 256 as
separate *matched* protocols on validation. A larger batch changes the
optimization trajectory, so apply the chosen batch size and learning-rate
search to every method. Do not shorten only SCR's rounds or epochs to make a
runtime comparison. For accuracy, use the declared rho, gamma and head-anchor
validation search in Step 3; never choose these values from test scores.

## Step 3: expand only after the pilot finishes

Pilot the shrinkage values `--rhos 0.25 0.5` and support scales
`--gammas 1 10` using validation. Keep `rho=1` represented by the original
`support_calibrated` arm, so it need not be repeated as `support_shrink`.
Include `--head-lambdas 0 0.01` to test whether the anchor contributes.
Use a common learning-rate grid and comparable tuning budget for the controls.
If the proposed arm receives more validation trials than a control, report
that difference rather than attributing the gain solely to the method.

Then repeat the selected study on BloodMNIST, DermaMNIST, OrganAMNIST, and
PathMNIST, each with the original Practical partition and the original
Dirichlet partition at the declared alpha values. Run one dataset and alpha
per output root. The matrix now accepts `--alpha 0.1` (or another declared
value) and `dermamnist`; neither option changes the partition algorithm.
Audit each Dirichlet alpha before a long run because some seeds may produce
an empty client.

For the final five-seed comparison, use seeds 42-46 for **every** arm. Run
`research.select` with exactly the grids used in the matrix, then
`research.finalize` once per validation lock and `research.analyze` on the
lock. `research.select` checks that the selected methods have identical
partition hashes for each seed. Final test data are loaded only by
`research.finalize` after validation selection.

Keep a stated success criterion before reading test results: paired-seed
macro-F1 is primary, with accuracy, every class's recall, worst-class
recall, calibration, runtime, and communication also reported. A single
pilot seed cannot support an accuracy or novelty claim. `baseline_bn` uses
shared BatchNorm state; it is not the FedBN algorithm.
