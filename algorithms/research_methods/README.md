# Algorithms for matched research experiments

The shared runner is `research.run` / `research.matrix`. Its algorithm choices
live here. Use `--methods` to select one or more choices; all selected methods
use the same data partitions, model choice, training budget and evaluation
pipeline. `research.matrix` runs them sequentially and produces comparison plots.

```powershell
& ..\.venv\Scripts\python.exe -m research.matrix --datasets pathmnist --partitions practical --models baseline baseline_bn --methods fedavg pooled_uniform_head support_calibrated --seeds 42 --lrs 0.001 --gammas 1 --head-lambdas 0.01 --clients 12 --rounds 30 --epochs 1 --batch-size 64 --optimizer adam --weight-decay 0 --device cuda --output-root results\research_scr_baseline --execute
```

Change `--methods` to switch algorithms. Run `python -m research.matrix
--list-methods` to see all available choices and their tuning axes. The
[SCR formulation](SCR.md) and its
[validation/test commands](../../research/SCR_BASELINE_EXPERIMENT.md) are documented separately.

| File | Responsibility |
| --- | --- |
| `__init__.py` | Method registry, client weight policies, tuning axes and local loss dispatch |
| `objectives.py` | One local minibatch loss function per method |
| `support.py` | SCR counts, class-client weights, head anchor and oracle aggregation weights |
| `models.py` | Model selection and classifier head lookup |

To add a research algorithm, write its local loss in `objectives.py` and
register it in `METHOD_REGISTRY` with its client weight policy and any tuning
axes. Existing axes (`gamma`, `head_lambda`, `fedlc_tau`, `fedvls_lambda`) are
shared by the matrix and validation selector. A new hyperparameter also needs
CLI arguments in `research/run.py`, `research/matrix.py`, and
`research/select.py`, plus a grid entry here. A method requiring different
server aggregation or client state needs a runner hook and a matched test.
Keep the test split out of development runs.

The Flower strategies exposed by `algorithms.get_strategy` are a separate
execution path. `algorithms/scr.py` now implements SCR there. It shares the
support formulas in this package, while the Flower training and evaluation
protocol differs from the matched research runner.
