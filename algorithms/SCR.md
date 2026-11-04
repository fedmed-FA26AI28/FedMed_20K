# SCR experiment: Dirichlet α = 0.1

This run compares support-calibrated risk (`scr`) with sample-weighted FedAvg on **the same 12-client Dirichlet split**. It uses the team's `baseline_bn` model on BloodMNIST, OrganAMNIST, and PathMNIST, seeds 42–46. The training settings match the earlier practical-partition SCR runs: 80 rounds, five local epochs, batch size 256, SGD at 0.01 with momentum 0.9, no weight decay, no augmentation, local test evaluation, `scr_gamma=1`, and `scr_head_lambda=0.01`. Both arms launch `client.client` workers concurrently.

The partition algorithm is unchanged. For each class, it draws 12 proportions from Dirichlet(0.1), assigns the official training samples to clients, and assigns official test samples with those same proportions. Each dataset and seed has one partition JSON shared by FedAvg and SCR. The official validation split is not partitioned by this runner.

## Kaggle notebook cells

First sync the current local files to the Kaggle repository directory. The public GitHub `main` does not yet contain this uncommitted runner or the empty-test evaluation fix. Enable a GPU and Internet for the notebook, then run these cells from the synced repository:

```python
%cd /kaggle/working/FedMed_20K
%pip install -q -r requirements.txt
```

```python
from pathlib import Path
import torch

runner = Path("scripts/run_scr_dirichlet01.py")
assert runner.exists(), "Sync the local runner to Kaggle first"
assert '"--client_module", "client.client"' in runner.read_text(encoding="utf-8")
assert torch.cuda.is_available(), "Enable a Kaggle GPU"
print(torch.cuda.get_device_name(0))
```

```python
!python -u -m scripts.run_scr_dirichlet01 --datasets bloodmnist --seeds 42 --dry-run
```

Run one seed first:

```python
!python -u -m scripts.run_scr_dirichlet01 --datasets bloodmnist --seeds 42
```

Then run the full matched matrix, or give one dataset and a few seeds per Kaggle session:

```python
!python -u -m scripts.run_scr_dirichlet01
```

The runner now defaults to `results/scr_bn_dirichlet01_parallel/` so completed runs made with the serialized `experiments.simulation_client` module cannot be skipped or mixed into the parallel-client summary. Twelve GPU workers can use substantially more memory at once; check the Kaggle GPU and host memory during the first pair before running more seeds.

To run SCR first and have your team run FedAvg separately, add `--strategies scr`. Use `--strategies fedavg` for the matched baseline. Both commands use the exact same partition JSON. The runner generates a missing partition before training and checks its metadata and client sample counts. BloodMNIST seed 44 and PathMNIST seed 46 can have a client with zero local test images; evaluation omits that client's zero-sample result from the client average while retaining its training data and the original split.

## Outputs and reruns

Each job writes to `results/scr_bn_dirichlet01_parallel/<dataset>_s<seed>_<strategy>/`. Its `experiment_manifest.json` records the exact command and partition hash. The per-job `simulation_report.md` and `*_compare_accuracy_curves.png` show training progress. The matrix runner also writes `results/scr_bn_dirichlet01_parallel/dirichlet01_summary.csv` and `dirichlet01_bta_by_seed.png` after each completed job. These matrix files collect all completed jobs in that output directory, including jobs from earlier Kaggle sessions using the parallel-client runner.

Rerunning the same command skips jobs with a completed 80-round result. If a Kaggle session stops during a job, the next invocation keeps the partial folder and writes a new `_retry1`, `_retry2`, and so on. Keep `results/scr_bn_dirichlet01_parallel/` in persistent storage or download it before the Kaggle session ends. This script does not commit or push results to Git.

The plotted BTA is the best pooled local-test accuracy across rounds, matching the team's earlier BTA reporting. It uses the test split for round selection, so use it for exploratory comparison with the earlier runs. A research claim needs a separate validation-selected checkpoint and one final test evaluation for each seed.
