# SCR on the team baseline CNN, with and without BatchNorm

Research methods are implemented in
[`algorithms/research_methods`](../algorithms/research_methods/README.md).
The [SCR formulation](../algorithms/research_methods/SCR.md) explains its
weights, anchor and limits.
Use `python -m research.matrix --list-methods` to see the available choices;
the `--methods` flag switches among them in the shared experiment runner.

Run these commands in PowerShell from `FedMed_20K`. The commands use the
repository's parent `.venv`, 12 sequential clients, zero weight decay, and no
training augmentation. Add `--augment` to **all** arms only for a separate
augmentation study.
If the MedMNIST archives are missing, add `--download` to `research.matrix`
once. `--device cuda` requires a working CUDA installation.

## 1. Preflight

```powershell
& ..\.venv\Scripts\python.exe -m unittest tests.test_support_calibrated -v
& ..\.venv\Scripts\python.exe -m research.audit_partitions --datasets pathmnist --seeds 42 --clients 12
```

## 2. Six-run pilot

This pilot checks that both architectures and all losses finish on the same
PathMNIST practical partition. It is one seed and does not establish a method
gain. The first command previews the six runs. Add `--execute` to train them.
Completed runs are skipped and interrupted runs resume from the last round.

```powershell
& ..\.venv\Scripts\python.exe -m research.matrix --datasets pathmnist --partitions practical --models baseline baseline_bn --methods fedavg pooled_uniform_head support_calibrated --seeds 42 --lrs 0.001 --gammas 1 --head-lambdas 0.01 --clients 12 --rounds 30 --epochs 1 --batch-size 64 --optimizer adam --weight-decay 0 --device cuda --output-root results\research_scr_baseline

& ..\.venv\Scripts\python.exe -m research.matrix --datasets pathmnist --partitions practical --models baseline baseline_bn --methods fedavg pooled_uniform_head support_calibrated --seeds 42 --lrs 0.001 --gammas 1 --head-lambdas 0.01 --clients 12 --rounds 30 --epochs 1 --batch-size 64 --optimizer adam --weight-decay 0 --device cuda --output-root results\research_scr_baseline --execute
```

After training, open the printed `validation_comparison.png` path under
`results/research_scr_baseline/plots/`. It compares the completed arms without
reading logs. Each run folder also contains `validation_curves.png` for
macro-F1, accuracy, worst-class recall, and training loss by round. The
adjacent `validation_summary.csv` contains the plot's values. These are
**validation** results, not final test comparisons. If a run crashes, rerun
the same matrix command; never treat an incomplete run as a zero score.

To regenerate these figures from existing completed runs without training:

```powershell
& ..\.venv\Scripts\python.exe -m research.plots --output-root results\research_scr_baseline --dataset pathmnist --partition practical
```

## 3. Paired five-seed validation study

Preview this 180-run matrix before executing. It compares FedAvg, the pooled
uniform-head control, and SCR on both models for seeds 42–46. Use the same
command with `--execute` only after the six-run pilot is stable. Run one
dataset and partition at a time. `bloodmnist` and `organamnist` can replace
`pathmnist`; `dirichlet` can replace `practical`.

```powershell
& ..\.venv\Scripts\python.exe -m research.matrix --datasets pathmnist --partitions practical --models baseline baseline_bn --methods fedavg pooled_uniform_head support_calibrated --seeds 42 43 44 45 46 --lrs 0.0003 0.001 --gammas 0.1 1 10 --head-lambdas 0 0.01 --clients 12 --rounds 30 --epochs 1 --batch-size 64 --optimizer adam --weight-decay 0 --device cuda --output-root results\research_scr_baseline
```

The other arms (`fedsld_literal`, `fedlc`, and `fedvls`) can be included in a
separate matched expansion. The printed FedSLD loss retains a batch sum and
may need an appropriate validation learning-rate range. Do not compare a
30-round Adam result directly to the paper's 80-round SGD BTA number.

## 4. Validation lock and one-time test

After **all** declared settings and five seeds finish, select the best
setting for each method using mean validation macro-F1. The commands below
must use exactly the same grids and protocol as the executed matrix.

```powershell
& ..\.venv\Scripts\python.exe -m research.select --dataset pathmnist --partition practical --model baseline --methods fedavg pooled_uniform_head support_calibrated --seeds 42 43 44 45 46 --lrs 0.0003 0.001 --gammas 0.1 1 10 --head-lambdas 0 0.01 --clients 12 --rounds 30 --epochs 1 --batch-size 64 --optimizer adam --weight-decay 0 --device cuda --output-root results\research_scr_baseline --lock-file results\research_scr_baseline\lock_pathmnist_practical_baseline.json

& ..\.venv\Scripts\python.exe -m research.select --dataset pathmnist --partition practical --model baseline_bn --methods fedavg pooled_uniform_head support_calibrated --seeds 42 43 44 45 46 --lrs 0.0003 0.001 --gammas 0.1 1 10 --head-lambdas 0 0.01 --clients 12 --rounds 30 --epochs 1 --batch-size 64 --optimizer adam --weight-decay 0 --device cuda --output-root results\research_scr_baseline --lock-file results\research_scr_baseline\lock_pathmnist_practical_baseline_bn.json
```

Evaluate each locked checkpoint on the official test split once, then create
the paired report:

```powershell
& ..\.venv\Scripts\python.exe -m research.finalize --lock-file results\research_scr_baseline\lock_pathmnist_practical_baseline.json --device cuda
& ..\.venv\Scripts\python.exe -m research.finalize --lock-file results\research_scr_baseline\lock_pathmnist_practical_baseline_bn.json --device cuda
& ..\.venv\Scripts\python.exe -m research.compare_models --baseline-lock results\research_scr_baseline\lock_pathmnist_practical_baseline.json --baseline-bn-lock results\research_scr_baseline\lock_pathmnist_practical_baseline_bn.json
```

The report compares SCR with FedAvg within each architecture, SCR with the
pooled control, and `baseline_bn` with `baseline` under SCR. The interaction
shows whether BatchNorm changes the incremental SCR gain. Report the paired
macro-F1 difference as primary, plus accuracy, every class's recall,
worst-class recall, calibration, runtime, and communication. Five simulated
seeds do not represent five independent hospitals. The comparison command
also writes a `.png` beside its Markdown and JSON reports with test macro-F1,
accuracy, worst-class recall, and SCR recall for every class.

## Paper-style protocol, kept separate

To study the paper's training budget, preview this separate six-run pilot.
It uses 80 rounds, five local epochs, SGD, no augmentation, and zero weight
decay for **every arm**. Add `--execute` after reviewing the planned runs.
This matches the broad training budget, but it is not by itself an exact
reproduction of the paper's full evaluation protocol.

```powershell
& ..\.venv\Scripts\python.exe -m research.matrix --datasets pathmnist --partitions practical --models baseline baseline_bn --methods fedavg pooled_uniform_head support_calibrated --seeds 42 --lrs 0.01 --gammas 1 --head-lambdas 0.01 --clients 12 --rounds 80 --epochs 5 --batch-size 256 --optimizer sgd --momentum 0.9 --weight-decay 0 --device cuda --output-root results\research_scr_paper_budget
```

Use a validation learning-rate search and five paired seeds before drawing
conclusions. Keep these results separate from the 30-round Adam study.

