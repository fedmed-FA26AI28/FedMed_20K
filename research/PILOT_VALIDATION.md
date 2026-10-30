# Validation pilot: support-calibrated risk

These are exploratory **validation** results, not final test results. No
`research.finalize` command has been run. All BloodMNIST rows below use the
official train/validation splits, Dirichlet \(\alpha=0.1\), seed 42, 12
sequential clients, 30 rounds, one local epoch, batch size 64, native 28-pixel
input, the shared GroupNorm CNN, local Adam at learning rate 0.001, and the
same train-only augmentation. Each value is the best validation macro-F1
across rounds for that single run. This is a development pilot and does not
constitute a five-seed, validation-selected comparison.

| Method | Setting | Best validation macro-F1 | Best-round worst-class recall |
|---|---|---:|---:|
| FedAvg | — | 0.5475 | 0.0000 |
| Printed-equation FedSLD | — | 0.5248 | 0.0000 |
| FedLC | best \(\tau=0.1\) in tested grid | 0.4304 | 0.0000 |
| Coverage-IW | head \(\lambda=0.01\) | 0.5903 | 0.0000 |
| FedSwitch aggregation oracle | exact histogram | 0.5896 | 0.0000 |
| Pooled class-balanced CE + uniform head | \(\lambda=0.01\) | 0.6201 | 0.0000 |
| Support risk, no head | \(\gamma=0.1\) | 0.6095 | 0.0000 |
| Support risk, no head | \(\gamma=1\) | 0.6246 | 0.0000 |
| Support risk, no head | \(\gamma=10\) | 0.6273 | 0.0000 |
| Support risk + uniform head | \(\gamma=1,\lambda=0.01\) | 0.6252 | 0.0000 |
| **Full support-calibrated method** | \(\gamma=1,\lambda=0.01\) | **0.6245** | **0.0000** |
| FedVLS adaptation | \(\lambda=0.01\) | **0.6495** | 0.0000 |
| FedVLS adaptation | \(\lambda=0.1\) | 0.6479 | 0.0414 |
| FedVLS adaptation | \(\lambda=1\) | 0.6191 | 0.0246 |

Representative development artifacts: [FedAvg](../results/research/bloodmnist_dirichlet_fedavg_s42_1389a500f490/development.json),
[pooled uniform-head control](../results/research/bloodmnist_dirichlet_pooled_uniform_head_s42_3c4466de763a/development.json),
[full support-calibrated method](../results/research/bloodmnist_dirichlet_support_calibrated_s42_88d19879fd93/development.json),
and [FedVLS adaptation](../results/research/bloodmnist_dirichlet_fedvls_s42_a68d2c8e4ba3/development.json).

The full method is 0.44 percentage points above the pooled uniform-head
control at this one setting, while replacing its coverage head with a uniform
head improves macro-F1 by 0.07 points. The best tested support-risk variant
is 0.73 points above the pooled control; the best FedVLS adaptation is 2.22
points above that support-risk variant. Thus this pilot gives **no evidence**
that coverage-dependent head protection adds value, and it does not yet
support a claim that the proposed method beats the strongest close comparator.
The zero worst-class recall for most arms is a material remaining problem.
The FedVLS adaptation shares this benchmark's backbone and optimizer rather
than reproducing the authors' full framework.

For the four linked BloodMNIST artifacts, recorded round times were 63.6 s
(FedAvg), 81.6 s (pooled uniform head), 78.8 s (full method), and 126.2 s
(FedVLS adaptation). Each has an estimated 55.2 MiB of model upload/download
traffic across the 30 rounds; this estimate excludes setup statistics and
transport overhead.

The [PathMNIST practical FedAvg development run](../results/research/pathmnist_practical_fedavg_s42_58bd56d50f59/development.json)
completed all 30 rounds with the same matched protocol, selecting round 29
by validation macro-F1: 0.8078 macro-F1, 0.8226 accuracy, and 0.4390
worst-class recall. The matched
[support-calibrated run](../results/research/pathmnist_practical_support_calibrated_s42_abd904db8261/development.json)
completed all 30 rounds and selected round 28: 0.8283 macro-F1, 0.8351
accuracy, and 0.6203 worst-class recall. That is 2.05 macro-F1 points and
18.13 worst-recall points above FedAvg for this single seed. The matched
[pooled class-balanced uniform-head control](../results/research/pathmnist_practical_pooled_uniform_head_s42_e1d1a907ebba/development.json)
also completed 30 rounds and selected round 28: 0.8204 macro-F1, 0.8291
accuracy, and 0.5399 worst-class recall. The support method is only **0.79
macro-F1 points** above this stronger control, though its worst-class recall
is 8.04 points higher. The three runs have the same partition hash. Their
wall times were 495 seconds (FedAvg), 677 seconds (pooled control), and 563
seconds (support). This verifies that the sequential runner can finish the
previously crashing practical PathMNIST
control; it is not numerically comparable to the paper's 80-round, five-epoch
best-test-accuracy result.

Across the BloodMNIST and PathMNIST single-seed settings completed here, the
full method's macro-F1 advantage over pooled class-balanced CE plus uniform
head protection is below one percentage point. The next decision must use
paired seeds and the locked test protocol; these pilot differences do not
establish a reliable gain or an A-rank-conference contribution.

The full study still requires all five seeds, both partitions, all three
datasets, shared validation tuning, and a locked one-time test evaluation.
The default matrix has 3,780 sequential development runs. Results and
checkpoints are saved in the ignored `results/research/` directory. Use
`research.select` only after the required development variants are complete.
