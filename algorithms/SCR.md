# Support-calibrated risk in the Flower simulation

`algorithms/scr.py` is the server strategy selected by `--strategies scr`.
The simulator passes its training partition JSON to the server. The server
reads each client's **training** class histogram, computes global class
counts and the support normalizer, then sends these aggregate statistics to
clients each round. A client computes its own fixed per-class weights and
anchors its classifier rows to the received global model. The server uses
sample-weighted FedAvg. Inference uses raw logits.

This is the same SCR weighting and head penalty as the
[method formulation](research_methods/SCR.md). It is a simulated partition
workflow with no formal privacy guarantee. The separate
[research runner](../research/README.md) uses validation-only checkpoint
selection; do not combine its scores with the Flower simulation's per-round
test metrics.

Run from the repository root. The following command compares FedAvg and SCR
on the same 12-client PathMNIST practical partition, using the team's
`baseline` model, zero weight decay, and serialized GPU training:

```bash
python -m experiments.run_simulation --config configs/simulation.yaml \
  --datasets pathmnist --partitions practical \
  --strategies fedavg scr --num_clients 12 --model baseline \
  --rounds 30 --local_epochs 1 --batch_size 64 \
  --optimizer adam --learning_rate 0.001 --weight_decay 0 \
  --scr_gamma 1 --scr_head_lambda 0.01 \
  --client_module experiments.simulation_client \
  --output_dir results/simulation_scr_baseline
```

For the BatchNorm arm, run the same command with `--model baseline_bn` and a
different `--output_dir`, such as `results/simulation_scr_baseline_bn`. The
simulator writes per-run curves and cross-strategy comparison plots under the
output directory. Its configured local test metric is useful for checking
that training works; use validation-selected checkpoints for research claims.

The simulation starts all 12 Python client processes before it serializes GPU
training. On the local Windows machine a 12-client smoke run failed during
PyTorch import with `WinError 1455` (paging file too small); a two-client
end-to-end run completed. Run the 12-client matrix on a VM with enough host
RAM and swap, or use the single-process [research runner](../research/README.md)
for limited-memory development. The GPU file lock alone does not reduce the
memory needed to start 12 processes.

The simulation requires partition JSON files with `class_distribution` for
every client. The launcher generates them automatically when absent. For a
manual server launch, pass `--partition_path` along with `--strategy scr`,
`--scr_gamma`, `--scr_head_lambda`, and `--min_clients`.
