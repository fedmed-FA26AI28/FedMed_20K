# Kaggle guide: 12-client FedSLD reconstruction

This guide runs the independent reconstruction of **FedSLD: Federated Learning
with Shared Label Distribution for Medical Image Classification** with the
paper's 12-client cross-silo setting.

The clients are simulated **sequentially** in one Python process. Each client
receives its own index partition, trains a private model copy, and returns only
model parameters and its sample count for sample-weighted FedAvg. All 12 client
models are not placed on the GPU at the same time. This makes the runner safer
for a Kaggle T4 than a 12-actor Ray simulation.

This is the paper-reproduction track. It deliberately evaluates the official
test set after each round and reports BTA, as the paper did. Do not use that
test-selection protocol for tuning the later proposed method.

## 0. Push the experiment branch first

The reproduction directory must exist on GitHub before Kaggle can clone it.
From the local `FedMed_20K` repository, confirm that it is committed and pushed
to the experiment branch:

```powershell
git switch experiment/vacant-class-distillation
git status
git add reproductions/FedSLD_2110_08378
git commit -m "Add reproducible 12-client FedSLD Kaggle runner"
git push origin experiment/vacant-class-distillation
```

Do not merge the branch into `main` for these experiments.

## 1. Create the Kaggle notebook

In the notebook settings:

1. Set **Accelerator** to **GPU T4 x2** or **GPU T4**. The runner uses one GPU.
2. Turn **Internet** on so MedMNIST can download its official archives.
3. Start with a fresh session.

Run this first cell:

```bash
!git clone --depth 1 --branch experiment/vacant-class-distillation \
  https://github.com/fedmed-FA26AI28/FedMed_20K.git
%cd /kaggle/working/FedMed_20K/reproductions/FedSLD_2110_08378
```

## 2. Install only the missing dependencies

Kaggle already provides CUDA-enabled PyTorch. Do not reinstall `torch` or
`torchvision` because that can replace the working CUDA build.

```bash
%pip install -q medmnist
%pip install -q -e . --no-deps
```

Verify the environment:

```bash
!python -c "import torch, medmnist; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('MedMNIST:', medmnist.__version__)"
```

`CUDA` must print `True` before using `--device cuda`.

## 3. Run the tests

```bash
!python -m unittest discover -s tests -v
```

All tests must pass before starting a long run.

Create the writable data and result directories. The runner also creates its
data root automatically, but this cell makes the Kaggle paths easy to verify:

```bash
!mkdir -p /kaggle/working/fedsld_data /kaggle/working/fedsld_results
```

## 4. Run a small 12-client smoke test

This uses all 12 simulated clients, but only one round and one local epoch:

```bash
!python -m fedsld_reproduction.run \
  --dataset organamnist \
  --algorithm fedsld \
  --partition practical \
  --num-clients 12 \
  --rounds 1 \
  --local-epochs 1 \
  --batch-size 256 \
  --learning-rate 0.01 \
  --device cuda \
  --data-root /kaggle/working/fedsld_data \
  --output-dir /kaggle/working/fedsld_results/smoke \
  --checkpoint-every 1
```

The beginning of the log should show `clients=12`, the 12 client partition
sizes, and `device=cuda`. At the end it should print paths for one JSON result,
the final checkpoint, the best-BTA checkpoint, and the resume checkpoint.

Inspect the generated files:

```bash
!find /kaggle/working/fedsld_results/smoke -maxdepth 5 -type f -printf '%p\n'
```

## 5. Run the first paper-scale comparison

Run FedAvg first, then FedSLD, with exactly the same dataset, partition, seed,
rounds, local epochs, batch size, and learning rate. The practical setting uses
80 rounds in the paper.

### OrganMNIST practical — FedAvg

```bash
!python -m fedsld_reproduction.run \
  --dataset organamnist \
  --algorithm fedavg \
  --partition practical \
  --num-clients 12 \
  --rounds 80 \
  --local-epochs 5 \
  --batch-size 256 \
  --learning-rate 0.01 \
  --seed 42 \
  --device cuda \
  --data-root /kaggle/working/fedsld_data \
  --output-dir /kaggle/working/fedsld_results/reference \
  --checkpoint-every 5
```

### OrganMNIST practical — FedSLD

```bash
!python -m fedsld_reproduction.run \
  --dataset organamnist \
  --algorithm fedsld \
  --partition practical \
  --num-clients 12 \
  --rounds 80 \
  --local-epochs 5 \
  --batch-size 256 \
  --learning-rate 0.01 \
  --seed 42 \
  --device cuda \
  --data-root /kaggle/working/fedsld_data \
  --output-dir /kaggle/working/fedsld_results/reference \
  --checkpoint-every 5
```

The final paper table reports practical OrganMNIST BTA of 85.69% for FedAvg and
84.75% for FedSLD. These are comparison targets, not pass/fail thresholds,
because the paper omitted several implementation details documented in the
main README. The arXiv-v1 table showing 64.99% and 66.13% is the pathological
setting and must not be compared with these practical runs.

## 6. Run the four 10-round reconstruction diagnostics

If the literal FedSLD result is unstable, do not start the remaining full
reference matrix. Run these four jobs on the same official split, partition,
seed, initialization procedure, and training budget. Only `--algorithm`
changes.

```bash
%%bash
COMMON="--dataset organamnist --partition practical --num-clients 12 --rounds 10 --local-epochs 5 --batch-size 256 --learning-rate 0.01 --seed 42 --device cuda --dataset-split official --data-root /kaggle/working/fedsld_data --output-dir /kaggle/working/fedsld_results/diagnostics --checkpoint-every 5"

python -m fedsld_reproduction.run --algorithm fedavg $COMMON
python -m fedsld_reproduction.run --algorithm fedsld $COMMON
python -m fedsld_reproduction.run --algorithm fedsld_weightnorm $COMMON
python -m fedsld_reproduction.run --algorithm fedsld_inverse $COMMON
```

In a Kaggle notebook, a multiline Python loop is more reliable than shell
variable handling across separate cells:

```python
import subprocess

common = [
    "--dataset", "organamnist",
    "--partition", "practical",
    "--num-clients", "12",
    "--rounds", "10",
    "--local-epochs", "5",
    "--batch-size", "256",
    "--learning-rate", "0.01",
    "--seed", "42",
    "--device", "cuda",
    "--dataset-split", "official",
    "--data-root", "/kaggle/working/fedsld_data",
    "--output-dir", "/kaggle/working/fedsld_results/diagnostics",
    "--checkpoint-every", "5",
]

for algorithm in (
    "fedavg",
    "fedsld",
    "fedsld_weightnorm",
    "fedsld_inverse",
):
    subprocess.run(
        ["python", "-m", "fedsld_reproduction.run", "--algorithm", algorithm, *common],
        check=True,
    )
```

The result JSON identifies the interpretations explicitly:

- `fedsld`: literal printed ratio `p_batch/p_global`, batch-size mean;
- `fedsld_weightnorm`: printed ratio, normalized by the sum of weights;
- `fedsld_inverse`: reversed `p_global/p_batch` ratio with weight normalization.

The last two are reconstruction hypotheses, not claims about the authors'
unreleased implementation.

Summarize the four JSON files after the loop finishes:

```bash
!python scripts/summarize_diagnostics.py /kaggle/working/fedsld_results/diagnostics
```

Compare BTA, final accuracy, the best round, and maximum training loss. Ten
rounds are a stability screen, not a paper-result reproduction.

After those four official-split runs, test the paper-style stratified split
only as a separate hypothesis. Start with FedAvg and the most stable FedSLD
variant, replacing `<algorithm>` below:

```bash
!python -m fedsld_reproduction.run \
  --dataset organamnist \
  --algorithm <algorithm> \
  --partition practical \
  --num-clients 12 \
  --rounds 10 \
  --local-epochs 5 \
  --batch-size 256 \
  --learning-rate 0.01 \
  --seed 42 \
  --device cuda \
  --dataset-split stratified \
  --data-root /kaggle/working/fedsld_data \
  --output-dir /kaggle/working/fedsld_results/diagnostics_stratified \
  --checkpoint-every 5
```

By default, stratified mode pools the official train, validation, and test
images, then creates a deterministic class-stratified split with the same total
test size as the official split. It is deliberately marked as a reconstruction
hypothesis in the JSON. It must not be mixed with official-split results.

## 7. Resume an interrupted run

Every five rounds, the runner writes:

```text
<output-dir>/<dataset>/<partition>/<algorithm>/seed42_resume.pt
```

If the current session is still alive, resume directly and write the next
checkpoint to the working directory:

```bash
!python -m fedsld_reproduction.run \
  --dataset organamnist \
  --algorithm fedsld \
  --partition practical \
  --num-clients 12 \
  --rounds 80 \
  --local-epochs 5 \
  --batch-size 256 \
  --learning-rate 0.01 \
  --seed 42 \
  --device cuda \
  --data-root /kaggle/working/fedsld_data \
  --output-dir /kaggle/working/fedsld_results/reference \
  --checkpoint-every 5 \
  --resume-from /kaggle/working/fedsld_results/reference/organamnist/practical/fedsld/seed42_resume.pt
```

To continue in a new Kaggle session:

1. Save the previous notebook as a version with output.
2. Attach that notebook output, or a dataset containing `seed42_resume.pt`, as
   a Kaggle input.
3. Pass its read-only `/kaggle/input/.../seed42_resume.pt` path to
   `--resume-from`.
4. Keep `--output-dir` under `/kaggle/working`; the new checkpoints must not be
   written into `/kaggle/input`.

All experiment-defining arguments must match the checkpoint. The runner rejects
a different seed, algorithm, partition, number of rounds, batch size, or local
training configuration.

## 8. Complete the reference matrix

After the first comparison works, the remaining reference experiments are:

| Dataset | Partition | Rounds | Algorithms |
|---|---:|---:|---|
| OrganMNIST | practical | 80 | FedAvg, FedProx, FedSLD |
| OrganMNIST | pathological | 160 | FedAvg, FedProx, FedSLD |
| PathMNIST | practical | 80 | FedAvg, FedProx, FedSLD |
| PathMNIST | pathological | 160 | FedAvg, FedProx, FedSLD |

Print all 12 commands without running them:

```bash
!python scripts/run_reference_matrix.py \
  --device cuda \
  --seed 42 \
  --output-dir /kaggle/working/fedsld_results/reference
```

Running all 12 jobs in one Kaggle session is not recommended. Run one job at a
time, save its output, and use the resume checkpoint when necessary. Start with
the two OrganMNIST practical jobs above before spending compute on the full
matrix.

## 9. Save and inspect results

List result JSON files:

```bash
!find /kaggle/working/fedsld_results -name '*.json' -printf '%p\n'
```

Create one archive for download or notebook output:

```bash
!cd /kaggle/working && zip -qr fedsld_results.zip fedsld_results
!ls -lh /kaggle/working/fedsld_results.zip
```

Use **Save Version** so everything under `/kaggle/working` becomes notebook
output. Preserve the JSON files and checkpoints; screenshots or copied console
logs alone are not sufficient research artifacts.

## Expected output layout

```text
/kaggle/working/fedsld_results/reference/
└── organamnist/
    └── practical/
        ├── fedavg/
        │   ├── seed42_<timestamp>.json
        │   ├── seed42_<timestamp>_final.pt
        │   ├── seed42_<timestamp>_best_bta.pt
        │   └── seed42_resume.pt
        └── fedsld/
            └── ...
```

The JSON records the full configuration, environment, per-client class counts,
partition hashes, per-round history, BTA, best round, and elapsed time.
