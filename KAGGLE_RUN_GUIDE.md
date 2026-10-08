# Run the full-training research study on Kaggle

Despite its name, this project uses all 11,959 images from the official
BloodMNIST training split, not 20,000. The commands below are **prepared but
not run on Kaggle**. Push this branch yourself, then clone it in a Kaggle
notebook (or attach it as an input). Enable a GPU accelerator for the GPU
commands. Internet access or an attached MedMNIST dataset is needed for
first-time package/data setup. Install `requirements-kaggle.txt` so the
notebook's PyTorch/torchvision build stays in place. Kaggle documents GPU
selection and saved `/kaggle/working` outputs in its
[notebook guide](https://www.kaggle.com/docs/notebooks).

From `/kaggle/working/FedMed_20K` (replace with your actual clone path), run
in separate notebook cells, prefixing shell commands with `!`:

```bash
python -m pip install -r requirements-kaggle.txt
python -c "import torch, flwr, ray, medmnist; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available(), 'Flower', flwr.__version__, 'Ray', ray.__version__)"
```

Optional two-round setup check (not a research result):

```bash
python -m experiments.run_simulation --strategy fedavg --num_clients 2 --alpha 0.3 --rounds 2 --local_epochs 1 --train_samples 160 --model tiny_cnn --size 28 --client_cpus 1 --ray_cpus 1 --ray_object_store_mb 256 --client_gpus 1 --output_dir results/kaggle_runs/smoke
```

First run a development-only centralized comparator for seed 42:

```bash
python -m experiments.run_centralized_research --epochs 30 --model tiny_cnn --size 64 --augment --seed 42 --batch_size 32 --learning_rate 0.001 --use_gpu --output_dir results/kaggle_runs/centralized
```

Then preview, and only when ready execute, the paired FL matrix for the same
seed and Dirichlet alpha 0.1. The matrix includes FedAvg, coverage,
distillation-only, coverage-plus-distillation, and the near-uniform-head
control. It uses ten **virtual** clients. `--ray_cpus 1 --client_cpus 1`
serializes the ten clients inside one Ray actor; it does **not** change their
partitions. `--client_gpus 1` gives that actor the GPU. The object-store cap
limits one source of Ray memory use.

```bash
python -m experiments.run_research_matrix distillation --seed 42 --alpha 0.1 --rounds 30 --local_epochs 1 --client_cpus 1 --ray_cpus 1 --ray_object_store_mb 256 --client_gpus 1 --output_dir results/kaggle_runs/fl
python -m experiments.run_research_matrix distillation --seed 42 --alpha 0.1 --rounds 30 --local_epochs 1 --client_cpus 1 --ray_cpus 1 --ray_object_store_mb 256 --client_gpus 1 --output_dir results/kaggle_runs/fl --execute
```

Repeat the FL command at `--alpha 0.3`, then repeat both alpha values at seeds
43 and 44. Run one central baseline per seed; alpha affects only the FL client
partition. Do not run the arms concurrently. If CUDA is unavailable, omit
`--use_gpu` and change `--client_gpus 1` to `--client_gpus 0` for both arms'
resource comparison. Centralized 30 epochs and FL 30 rounds with one local
epoch both make 30 passes over their permitted training pool, but their
optimizer states and computation order differ.

Development runs save `run_spec.json` and final-model **validation** metrics.
They do not load test. Compare paired-seed macro-F1, worst-class/per-class
recall, calibration, runtime, and model/communication bytes.

## Diagnose a weak FedAvg baseline before claiming a method gain

The seed-42, alpha-0.1 `tiny_cnn` FedAvg run completed 30 rounds but its
final validation macro-F1 was 0.236, versus 0.945 for the centralized CNN.
This is a development result, **not** a test score or proof of the cause.
`tiny_cnn` uses BatchNorm, including client-dependent running statistics.
`tiny_cnn_gn` keeps the same 32/64-channel convolutional architecture but
replaces its two BatchNorm layers with GroupNorm, which has no running
statistics to aggregate. This is a diagnostic baseline, not a proposed novel
method or a guaranteed improvement. GroupNorm uses input statistics in both
training and evaluation ([PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html)).

After committing and pushing this experiment-branch update, pull it on
Kaggle and run these **one at a time**. First vary only the partition skew,
using the original CNN:

```bash
python -m experiments.run_simulation --strategy fedavg --num_clients 10 --alpha 0.3 --rounds 30 --local_epochs 1 --batch_size 32 --learning_rate 0.001 --model tiny_cnn --size 64 --augment --seed 42 --client_cpus 1 --ray_cpus 1 --ray_object_store_mb 256 --client_gpus 1 --output_dir results/kaggle_runs/diagnostics
```

Then compare the GroupNorm model with a matching centralized run and the same
severe alpha-0.1 FL setup:

```bash
python -m experiments.run_centralized_research --epochs 30 --model tiny_cnn_gn --size 64 --augment --seed 42 --batch_size 32 --learning_rate 0.001 --use_gpu --output_dir results/kaggle_runs/diagnostics/centralized_gn
python -m experiments.run_simulation --strategy fedavg --num_clients 10 --alpha 0.1 --rounds 30 --local_epochs 1 --batch_size 32 --learning_rate 0.001 --model tiny_cnn_gn --size 64 --augment --seed 42 --client_cpus 1 --ray_cpus 1 --ray_object_store_mb 256 --client_gpus 1 --output_dir results/kaggle_runs/diagnostics
```

Compare **final validation macro-F1 and per-class recall** for (a) BatchNorm
FedAvg at alpha 0.1 versus 0.3, and (b) GroupNorm versus BatchNorm at alpha
0.1, each alongside its own centralized comparator. Keep the train budget,
seed, rounds, local epochs, and validation split fixed. A single seed is only
a diagnostic; repeat promising comparisons at seeds 43 and 44 before making
research claims. If GroupNorm is useful, the existing matrix accepts
`--model tiny_cnn_gn` for matched method comparisons. Do **not** use
`--final_test` during these diagnostics.

Once this diagnostic is resolved, freeze the configuration using validation
before a locked final run:

```bash
python -m experiments.run_centralized_research --final_test --locked_config results/kaggle_runs/centralized/SEED_RUN/run_spec.json --output_dir results/kaggle_runs/final_centralized
python -m experiments.run_simulation --final_test --locked_config results/kaggle_runs/fl/STRATEGY/SEED_RUN/run_spec.json --output_dir results/kaggle_runs/final_fl
```

Replace the example spec paths with the actual generated paths. Each final
command retrains from its frozen settings and evaluates the official test split
once. Do not choose hyperparameters or models using test scores. Keep the
saved `/kaggle/working` outputs before ending the notebook session; the
`results/kaggle_runs/` directory is intentionally Git-ignored. The earlier
local 20K pilot predates the model-initialization seed correction; do not mix
its scores into the matched Kaggle analysis.
