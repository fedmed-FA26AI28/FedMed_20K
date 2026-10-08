# Optional vacant-class distillation experiment

This branch adds two opt-in simulation/real-server strategies without changing
FedAvg, coverage training, partitions, or final-test handling:

- `vacant_distill`: raw cross-entropy plus a frozen global-teacher loss.
- `coverage_distill`: the existing coverage objective plus the same teacher loss.

This is **FedVLS-inspired**, not a FedVLS reproduction: it does not implement
FedVLS logit suppression. For each client class with at most
`--distill_max_count` local **training** examples (default: zero), the student
matches the received global model's prediction probability. The remaining
classes are grouped into one comparison bin, so a single vacant class still
has a nonzero learning signal. The teacher is copied after loading the global
round model, frozen, and discarded after local fit. It uses the same local
training images, never validation/test images; no teacher or labels are sent
back. Server aggregation remains sample-weighted FedAvg.

Default `--distill_warmup_rounds 1` skips distillation in round 1, when the
teacher is random. `--distill_mu 0.1` and `--distill_temperature 2` are **pilot
values**, not tuned or guaranteed optimal. If a client has no qualifying class,
the teacher is not created and its distillation loss is zero. Telemetry reports
`distill_loss` and `distill_active_classes`. Expect an extra model copy and
teacher forward pass on active clients; benchmark memory and time on Orin/Nano.

From this project directory, run a small validation-only smoke first:

```powershell
python -m experiments.run_simulation --strategy vacant_distill --num_clients 2 --rounds 2 --train_samples 160 --model tiny_cnn --size 28 --alpha 0.3 --distill_max_count 10
```

Then list or run the paired study. It uses identical seeds, train budgets,
partitions, rounds, local epochs, model, resolution, and augmentation for
FedAvg, coverage, distillation alone, and coverage plus distillation. A fifth
arm uses `--coverage_kappa 1e12` as a near-uniform-head control; it is not a
penalty-strength-matched control and should not be used to claim a causal
benefit for coverage weighting on its own.

```powershell
python -m experiments.run_research_matrix distillation
python -m experiments.run_research_matrix distillation --execute --rounds 30 --local_epochs 1
```

Use validation macro-F1 and per-class/worst-class recall for model selection.
Development runs do **not** load test. Only after fixing the configuration,
run the locked final evaluation; do not retune based on test:

```powershell
python -m experiments.run_simulation --final_test --locked_config path/to/development/run_spec.json
```

A 5K-vs-full comparison needs the corresponding 5K branch and the same selected
settings. BloodMNIST's synthetic clients are not independent hospitals.

On each Jetson, compare the same client shard and sample cap with the train-only
benchmark. Use distinct output files, and check `teacher_model_copy` and
`distill_active_classes` in the JSON; otherwise the proposed loss was inactive:

```bash
python -m experiments.benchmark_jetson --strategy fedavg --client_id 0 --output results/jetson_fedavg.json
python -m experiments.benchmark_jetson --strategy vacant_distill --client_id 0 --output results/jetson_distill.json
```

This Git branch is `experiment/vacant-class-distillation`. The branch
`baseline/pre-distillation` retains the exact pre-experiment working baseline.
After the worktree is clean, `git switch baseline/pre-distillation` removes
the experimental method from your active files without deleting its branch.
`main` remains untouched and may be older than this preserved baseline.

Related paper: [FedVLS, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33864).
