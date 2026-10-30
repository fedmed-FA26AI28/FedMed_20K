# Support-calibrated risk: GroupNorm and BatchNorm experiment

Run commands from the clean `FedMed_20K_scr_experiment` worktree in
PowerShell. It is based on confirmed upstream `main` commit `2351b326`;
the original `FedMed_20K` checkout remains on `9e9da2b7` with local changes.
Here **SCR** means the
support-calibrated risk in `research/support_calibrated.py`, selected with
`--method support_calibrated`. The team's `algorithms/scr.py` at upstream
commit `2351b326` is still a strategy skeleton that delegates to baseline
aggregation; `--strategies scr` in `experiments.run_simulation` does **not**
run this proposed loss.

The research runner trains 12 simulated clients sequentially, one GPU model
at a time. It uses official MedMNIST train for optimization, validation for
checkpoint and hyperparameter selection, and the official test set once after
selection. The four primary arms form a matched 2×2 comparison:

| Normalization | Ordinary FedAvg loss | Support-calibrated risk |
|---|---|---|
| GroupNorm | `gn` + `fedavg` | `gn` + `support_calibrated` |
| BatchNorm | `bn` + `fedavg` | `bn` + `support_calibrated` |

Both models have the same convolutions, classifier, and initial convolution
and classifier weights. Only the two normalization modules differ. BN affine
parameters and running statistics participate in FedAvg; this is global BN,
not FedBN's client-local BN. The paper-style `baseline` and `baseline_bn`
models in the team's latest code are a separate, much larger architecture and
should not be mixed into this 2×2 comparison.
Training-mode BN couples the examples in a minibatch, so the method's
per-example one-step risk identity should not be claimed as an exact identity
for BN training. The BN arm tests whether the same weighting and anchor work
empirically under that normalization.

All four arms must use the same dataset, partition seed, augmentation, 12
clients, 30 rounds, one local epoch, batch size, optimizer, learning-rate grid,
and **zero weight decay**. This is the new matched benchmark. It is distinct
from the paper's 80-round, five-epoch BTA reproduction. The team's current
`configs/simulation.yaml` specifies `weight_decay: 1e-4`; the commands below
explicitly pass `--weight-decay 0` to the research runner.

## 1. Check the environment

```powershell
& ..\.venv\Scripts\python.exe -m unittest tests.test_support_calibrated -v
& ..\.venv\Scripts\python.exe -m research.audit_partitions
```

If MedMNIST is not cached, add `--download` to `research.matrix` once. The
development runner does not load test data. `--device cuda` requires CUDA;
use `--device cpu` only for a short smoke test. Do not use
`--limit-per-client` for reported results.

## 2. Four-arm full-data pilot

Start with BloodMNIST practical, seed 42, one fixed setting. Run the preview
first. Add `--execute` to the same command to train all four arms:

```powershell
& ..\.venv\Scripts\python.exe -m research.matrix --datasets bloodmnist --partitions practical --norms gn bn --methods fedavg support_calibrated --seeds 42 --lrs 0.001 --gammas 1 --head-lambdas 0.01 --clients 12 --rounds 30 --epochs 1 --batch-size 64 --optimizer adam --weight-decay 0 --device cuda --output-root results\research_scr_norm
& ..\.venv\Scripts\python.exe -m research.matrix --datasets bloodmnist --partitions practical --norms gn bn --methods fedavg support_calibrated --seeds 42 --lrs 0.001 --gammas 1 --head-lambdas 0.01 --clients 12 --rounds 30 --epochs 1 --batch-size 64 --optimizer adam --weight-decay 0 --device cuda --output-root results\research_scr_norm --execute
```

Read each `development.json` and check completion, finite losses, validation
macro-F1, per-class recall, and runtime. A one-seed pilot is a stability check,
not evidence that one method is better. `research.matrix --execute` skips
completed runs, and a partially completed run resumes from its last round.

## 3. Main validation sweep

For one dataset and partition, use seeds 42–46 and a shared learning-rate
grid. Include `pooled_uniform_head`: it checks whether SCR's class–client risk
adds value beyond class balancing plus a uniform classifier penalty.
Preview this matrix before executing. The command below plans **390
sequential development runs** for PathMNIST practical and can take substantial
time on a laptop GPU. Run one dataset and partition at a time.

```powershell
& ..\.venv\Scripts\python.exe -m research.matrix --datasets pathmnist --partitions practical --norms gn bn --methods fedavg pooled_uniform_head support_calibrated --seeds 42 43 44 45 46 --lrs 0.0001 0.0003 0.001 --gammas 0.1 1 10 --head-lambdas 0 0.01 0.1 --clients 12 --rounds 30 --epochs 1 --batch-size 64 --optimizer adam --weight-decay 0 --device cuda --output-root results\research_scr_norm
```

Append `--execute` when ready to run. Repeat the same protocol for
BloodMNIST and OrganAMNIST, then for Dirichlet partitioning if compute
permits. For a smaller first study, narrow the declared grids before any test
evaluation and use those same grids in the selection commands.

## 4. Lock validation choices and evaluate test once

After **all** declared settings and seeds finish, select the best setting for
each arm by mean validation macro-F1. `research.select` also records each
seed's best validation round. These two commands lock GN and BN separately:

```powershell
& ..\.venv\Scripts\python.exe -m research.select --dataset pathmnist --partition practical --norm gn --methods fedavg pooled_uniform_head support_calibrated --seeds 42 43 44 45 46 --lrs 0.0001 0.0003 0.001 --gammas 0.1 1 10 --head-lambdas 0 0.01 0.1 --clients 12 --rounds 30 --epochs 1 --batch-size 64 --optimizer adam --weight-decay 0 --device cuda --output-root results\research_scr_norm --lock-file results\research_scr_norm\lock_pathmnist_practical_gn.json
& ..\.venv\Scripts\python.exe -m research.select --dataset pathmnist --partition practical --norm bn --methods fedavg pooled_uniform_head support_calibrated --seeds 42 43 44 45 46 --lrs 0.0001 0.0003 0.001 --gammas 0.1 1 10 --head-lambdas 0 0.01 0.1 --clients 12 --rounds 30 --epochs 1 --batch-size 64 --optimizer adam --weight-decay 0 --device cuda --output-root results\research_scr_norm --lock-file results\research_scr_norm\lock_pathmnist_practical_bn.json
```

Then run the locked checkpoints on test once and build the paired report:

```powershell
& ..\.venv\Scripts\python.exe -m research.finalize --lock-file results\research_scr_norm\lock_pathmnist_practical_gn.json --device cuda
& ..\.venv\Scripts\python.exe -m research.finalize --lock-file results\research_scr_norm\lock_pathmnist_practical_bn.json --device cuda
& ..\.venv\Scripts\python.exe -m research.compare_norm --gn-lock results\research_scr_norm\lock_pathmnist_practical_gn.json --bn-lock results\research_scr_norm\lock_pathmnist_practical_bn.json
```

The comparison is written to
`results/research_scr_norm/lock_pathmnist_practical_gn_vs_bn.md` and JSON.
The report shows SCR minus FedAvg within GN and BN, BN minus GN under SCR,
and the difference between the two SCR gains. The JSON also contains
per-seed differences and paired bootstrap intervals.

## Reading the result

- Claim an SCR gain only if its paired macro-F1 improvement over FedAvg is
  consistent across seeds. Compare SCR with `pooled_uniform_head` as the
  essential mechanism control.
- Claim BN helps SCR only if BN+SCR beats GN+SCR on paired seeds. The
  interaction checks whether BN changes the **incremental** effect of SCR.
- Report accuracy, every class's recall, worst-class recall, calibration,
  runtime, and communication. A macro-F1 gain with a severe loss on a critical
  class needs explicit discussion.
- Five simulated seeds estimate sensitivity to partitions and initialization;
  they do not represent five independent hospitals. Results on these
  partitions cannot establish cross-hospital scanner robustness.
- A failed or incomplete run is not a zero score. Resolve it and rerun before
  locking settings. Do not select settings using test results.

The original printed FedSLD loss, FedLC, FedVLS, Coverage-IW, and other
controls are available as `--methods` in `research.matrix`; use matched
normalization and protocol when expanding beyond this first 2×2 study.
