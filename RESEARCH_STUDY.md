# FedMed full-train research workflow

Despite the folder name, this project uses the official 11,959-image BloodMNIST **training** split. It classifies eight normal blood-cell types, not disease. Virtual clients receive indices into separate training partitions; only weights, sample counts, and metrics are exchanged during FL. The official test split is reserved for locked final runs.

From this project directory:

```powershell
python -m unittest discover -s tests -p 'test*.py'
python -m experiments.verify_resolution_alignment
python -m experiments.research_eda --size 64 --num_clients 10 --alpha 0.1
python -m experiments.run_research_matrix main
python -m experiments.run_research_matrix main --execute --rounds 30 --local_epochs 1
python -m experiments.run_research_matrix ablation --execute --rounds 30 --local_epochs 1
```

The matrix command prints jobs by default. Add `--client_gpus 0.5` only if CUDA works and Ray can share your GPU. Main jobs compare FedAvg, FedProx, client-local balanced sampling, and coverage-aware training at α=0.1/0.3 across seeds 42/43/44. Ablations test coverage's logit adjustment and head penalty separately at α=0.1. All use the same TinyCNN, 64×64 input, mild training-only augmentation, ten clients, one local epoch, and 30 rounds. `tiny_cnn` matches the 5K project's `tiny_cnn` exactly. `mobilenet_v3_small` is also available for model/edge-cost comparisons.

These are **development** runs: results contain `final_validation_metrics` (exact pooled monitor-set macro-F1, worst-class recall, calibration), a `run_spec.json`, and no final-test file. Select and freeze configurations using validation. For each locked run, use its spec to perform one final evaluation:

```powershell
python -m experiments.run_simulation --final_test --locked_config path/to/development/run_spec.json
```

The final command reloads the frozen settings, retrains, calibrates on the separate validation-calibration subset, then evaluates test once. Do not use test results to change a configuration. A different seed needs its own development and final run spec.

On each physical Jetson, benchmark using the **training** split only:

Use `requirements-jetson.txt` for supporting packages after installing a PyTorch/torchvision build compatible with that Jetson's JetPack; the PC `requirements.txt` is not a Jetson wheel recipe.

```bash
python -m experiments.benchmark_jetson --model tiny_cnn --size 64 --client_id 0 --output results/orin_tiny64.json
python -m experiments.benchmark_jetson --model tiny_cnn --size 64 --client_id 1 --output results/nano_tiny64.json
```

Run each command on its respective device; also benchmark `mobilenet_v3_small`. The benchmark records batch-one p50/p95 inference latency, local training time for up to 256 assigned samples, process/CUDA memory, and parameter bytes. Record JetPack, power mode, and clock settings alongside results. Physical benchmarks have **not** been run by the implementation agent.

For a two-device FL check, generate a two-client training partition with `python -m experiments.run_partition --num_clients 2 --alpha 0.3`, start `python -m server.server --min_clients 2 --rounds 2 --model tiny_cnn --size 64` on the server, and start `python -m client.client --server_address SERVER_IP:8080 --client_id 0 --num_clients 2 --model tiny_cnn --size 64` and the same command with `--client_id 1` on Orin and Nano. Pass the same generated partition path to both with `--partition_path`, and set each `--device_type` as appropriate. Physical runs are validation-only by default and save the final aggregated `global_model.pt`; do not add `--final_test` to this integration check. Both devices must have the public benchmark and partition manifest locally before FL; no images are transmitted by Flower rounds. This is **not** evidence of genuinely private hospital silos.
