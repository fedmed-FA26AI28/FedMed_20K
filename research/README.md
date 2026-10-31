# Matched research experiments

This directory contains the shared experiment runner, data handling,
validation-only selection, test finalization and plots. Implementations and a
method registry live in [`algorithms/research_methods`](../algorithms/research_methods/README.md).
Use `python -m research.matrix --list-methods` to see the algorithm choices,
then choose arms with `--methods`. Run commands from the repository root.

The runner uses official MedMNIST train/validation/test splits and trains 12
simulated clients sequentially, with one model on the GPU at a time. It
defaults to zero weight decay and no augmentation. Choose a model with
`--models baseline baseline_bn` and use the same partition, seed, optimizer,
rounds, epochs and search budget for matched arms.

## Experiment flow

1. Preview and run a matrix with `python -m research.matrix --methods ...`;
   add `--execute` to train. Completed runs are skipped, and an interrupted
   run resumes from its last saved round.
2. Inspect each `validation_curves.png` and the matrix's
   `validation_comparison.png` and `validation_summary.csv` under the output
   root. These show development validation results.
3. After all declared settings and seeds finish, use `research.select` to
   lock hyperparameters by mean validation macro-F1. `research.run` and
   `research.select` never load the test split.
4. Use `research.finalize` to evaluate each locked checkpoint once on the
   official test split, then `research.compare_models` for paired results and
   a test comparison plot.

The proposed SCR formulation and controls are described in
[`algorithms/research_methods/SCR.md`](../algorithms/research_methods/SCR.md).
Its copyable pilot and five-seed commands are in
[SCR_BASELINE_EXPERIMENT.md](SCR_BASELINE_EXPERIMENT.md). The matched 30-round
study is separate from the 80-round FedSLD paper-budget reproduction.

The Flower strategies selected with `--strategies` are a separate execution
path. `--strategies scr` runs SCR through the team's Flower server and clients;
see the [Flower SCR guide](../algorithms/SCR.md). The two runners share the
support weight and anchor formulas but have different training and evaluation
protocols. Keep their results separate.
The team's CLA and FedBNS Flower implementations are not registered in this
research runner; their client and aggregation states require their own matched
research adapters before they can be compared in this matrix.
