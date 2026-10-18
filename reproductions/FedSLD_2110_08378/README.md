# FedSLD reference reproduction

This directory is an isolated subproject reconstructing **FedSLD: Federated
Learning with Shared Label Distribution for Medical Image Classification**
(Luo and Wu, ISBI 2022; arXiv:2110.08378v1). It lives under the experiment
repository so it can be versioned on the same branch, but it does not import
the FedMed implementation or share its result directories.

For a memory-safe 12-client Kaggle workflow, including smoke testing and
interruption-safe resume instructions, see [KAGGLE_RUN_GUIDE.md](KAGGLE_RUN_GUIDE.md).

The study has three stages:

1. Reproduce the paper on OrganMNIST(Axial) and PathMNIST.
2. Port FedAvg, FedProx, and FedSLD unchanged to BloodMNIST using one matched
   protocol.
3. Compare the unchanged reference methods with the FedMed proposed method.

Do not call stage 2 a reproduction of the paper: BloodMNIST was not evaluated
in the FedSLD paper.

## Paper protocol implemented here

| Item | Reconstruction |
|---|---|
| Clients | 12, with full participation each round |
| Model | Four-layer CNN: two 5x5 convolution layers and two linear layers; hidden linear layer has 500 units |
| Batch size | 256 |
| Local epochs | 5 |
| Learning rate | 0.01 |
| Practical non-IID | For every class: ten 1% shards, one 10% shard, and one 80% shard, randomly assigned to the 12 clients |
| Pathological non-IID | Every client is assigned two classes with random per-class allocation |
| Practical rounds | 80 |
| Pathological rounds | 160 |
| Aggregation | Sample-weighted FedAvg |
| Study metric | Best Test Accuracy (BTA) over rounds |

FedSLD shares client class counts, estimates the federation prior, and weights
the cross-entropy for sample `y` by
`batch_frequency[y] / federation_prior[y]`.

## Explicit reconstruction assumptions

The paper and public slides do not provide source code, random seeds, optimizer
type, convolution channel widths, pooling/activation details, preprocessing, a
FedProx coefficient, or the exact handling of MedMNIST's validation split.
Consequently, this implementation uses:

- the common LeNet/PySyft-style widths 20 and 50, ReLU, and 2x2 max-pooling;
- SGD without momentum or weight decay;
- pixel conversion to `[0, 1]` without augmentation;
- official `train + val` as the paper's training pool and official `test` as
  test, because the paper describes only train/test;
- `mu=0.01` for the FedProx comparator, configurable from the CLI;
- mean reduction for the weighted minibatch loss. The paper prints a sum, but
  does not document its code-level reduction.

These assumptions must be reported with the results. Matching the published
numbers exactly is not guaranteed. A faithful claim is **independent
reconstruction**, followed by an empirical reproduction check.

The paper evaluates test data after every round and reports BTA, the highest
combined test accuracy over all rounds. Although the paper also reports BMCTA,
this reconstruction intentionally uses only BTA. The runner marks the repeated
test evaluation in every result file. This is not the protocol for our improved
study: configuration selection there must use validation, followed by one
locked test evaluation.

## Setup

From this directory:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Small pipeline check:

```bash
python -m fedsld_reproduction.run \
  --dataset organamnist \
  --algorithm fedsld \
  --partition practical \
  --rounds 1 \
  --local-epochs 1 \
  --device cuda \
  --output-dir results/smoke
```

Paper-scale runs save a resumable checkpoint every five rounds by default. Use
`--resume-from path/to/seed42_resume.pt` to continue the exact same experiment,
or `--checkpoint-every 0` to disable progress checkpoints.

Paper-scale examples:

```bash
python -m fedsld_reproduction.run --dataset organamnist --algorithm fedavg  --partition practical --device cuda
python -m fedsld_reproduction.run --dataset organamnist --algorithm fedprox --partition practical --device cuda
python -m fedsld_reproduction.run --dataset organamnist --algorithm fedsld  --partition practical --device cuda

python -m fedsld_reproduction.run --dataset pathmnist --algorithm fedavg --partition pathological --device cuda
python -m fedsld_reproduction.run --dataset pathmnist --algorithm fedsld --partition pathological --device cuda
```

The dry-run matrix command prints every reference job without starting it:

```bash
python scripts/run_reference_matrix.py
```

Add `--execute` only on Kaggle after the smoke run succeeds.

## Published accuracy targets

These are diagnostic targets, not acceptance thresholds:

| Dataset/partition | Method | BTA |
|---|---|---:|
| OrganMNIST/pathological | FedAvg | 64.99 |
| OrganMNIST/pathological | FedProx | 65.10 |
| OrganMNIST/pathological | FedSLD | 66.13 |
| PathMNIST/pathological | FedAvg | 57.54 |
| PathMNIST/pathological | FedProx | 57.56 |
| PathMNIST/pathological | FedSLD | 59.11 |
| OrganMNIST/practical | FedAvg | 85.69 |
| OrganMNIST/practical | FedProx | 85.54 |
| OrganMNIST/practical | FedSLD | 87.37 |
| PathMNIST/practical | FedAvg | 57.38 |
| PathMNIST/practical | FedProx | 57.72 |
| PathMNIST/practical | FedSLD | 57.90 |

## BloodMNIST reference port

After validating the reconstruction, run the unchanged algorithms on the same
BloodMNIST protocol:

```bash
python -m fedsld_reproduction.run --dataset bloodmnist --algorithm fedavg --partition practical --device cuda
python -m fedsld_reproduction.run --dataset bloodmnist --algorithm fedsld --partition practical --device cuda
```

For the final research comparison, use the exact same BloodMNIST indices,
partitions, initialization, rounds, local steps, and evaluation policy across
FedSLD and the proposed method. That matched comparison belongs in the FedMed
experiment framework, not in the paper-reproduction results table.

