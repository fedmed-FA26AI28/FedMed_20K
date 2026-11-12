# Investigating a run with the experiment monitor

Open **Training**, select a session, and scroll to **Experiment monitor**. The coordinator receives measurements from every participating institution. An Institution portal shows only its own run records. This monitor runs locally; no W&B account, cloud service, or internet connection is needed during a prepared LAN demonstration.

## What to look at

| View | Questions it helps answer |
|---|---|
| Learning curves | Is test performance improving? Is training improving while test performance stalls? Are gradients or model updates growing sharply? |
| Client diagnostics | Which institution is slow or has poor test performance? How many samples does it have? Does SCR heavily weight classes with little support? |
| Class performance | Which true classes are confused with which predicted classes? Is overall accuracy hiding low recall for less frequent classes? |
| System & timing | Is time spent in local training, round coordination, or evaluation? Are memory usage and fit-time differences increasing? |
| Live logs | What did each training process actually print? Did a warning, traceback, or disconnect explain a failed round? |
| Configuration | Which exact hyperparameters, roster, dataset checksum, and partition checksum produced this run? |

Use **Source** to inspect an institution, **Round** for class/client details, and **Pause** to stop screen polling while you examine a result. Pausing the monitor does not pause training. Refreshing the browser preserves the recorded run.

Round-level learning curves and headline metrics describe the session. The source filter selects client/epoch, class, resource, and log records. On an Institution portal, the session statistics represent that institution's local evaluation, not global accuracy.

In **Learning curves → Compare with another run**, overlay test accuracy, test cross-entropy, and full-round time. The comparison checks recorded dataset/partition hashes, roster mapping, model, seed, training budget, and shared optimizer settings. Missing partition identity is explicitly unverified. A matching configuration does not establish identical hardware, runtime load, or statistical significance; strategy-specific settings are intentionally allowed to differ.

## Metric definitions

- **Current test accuracy:** most recent available completed-round test evaluation. On the coordinator this is the combined local test accuracy recorded by the research implementation.
- **Final test accuracy:** current accuracy only after the run is completed. Failed, cancelled, or still-running sessions have no final accuracy.
- **BTA:** maximum test accuracy over recorded completed rounds, including a legitimate value of zero. **BMCTA:** maximum mean client test accuracy. Both select a test round and are exploratory; use validation-selected checkpoints and a separate final test evaluation for a paper claim.
- **Training objective:** sample-weighted average of clients' final local-epoch objectives for round curves. Epoch curves show each client's own objective. **Test loss:** ordinary cross-entropy of the evaluated model. Under SCR these objectives have different meanings; do not interpret their numerical gap as an ordinary train/test CE gap.
- **SCR diagnostics:** unweighted raw-logit cross-entropy, weighted cross-entropy before anchoring, and the additive head penalty. Each is averaged over samples seen during that epoch. The support table exposes the fixed training class counts, sample weights, and anchor coefficients used by the method.
- **Gradient L2:** gradient norm of the last minibatch in an epoch, after backward and before any gradient clearing. It is neither a gradient histogram nor the mean gradient norm over an epoch. For SCAFFOLD, the decoupled drift correction is not part of this gradient norm.
- **Update L2:** Euclidean norm of the difference between the client's starting and ending floating model-state arrays, including floating BatchNorm buffers. The starting state preserves local BatchNorm for FedBN/FedBNS. **Relative update:** update L2 divided by starting-state L2; undefined for zero starting norm. These are diagnostics, not a proof of harmful client drift.
- **Class metrics:** confusion counts use raw-logit argmax, with true classes as rows and predicted classes as columns. Counts are summed across the selected reporting institutions before calculating metrics. FedBN/FedBNS therefore pool predictions from different institution-specific models. Precision is undefined with no predicted positives; recall is undefined with no true support. Macro F1 and balanced accuracy include only classes with true test support. Empty test sets are skipped, never shown as measured zero accuracy.
- **Fit / evaluation / full round time:** server-observed elapsed time. Fit includes client waiting, local training, and aggregation; full round time includes evaluation. Client fit time measures the local training loop. The fit-time spread is slowest minus fastest local training time, not measured network latency.
- **Payload KiB:** uncompressed parameter-array sizes during fit, measured at clients. SCAFFOLD control arrays are included. These figures exclude Flower serialization, protocol overhead, evaluation transfers, and monitoring traffic, so they are not total wire traffic. Reports can arrive late; the round CSV includes reporting-client counts.
- **CPU / RAM:** host-wide samples about every five seconds. **Process RSS MiB:** sum over the owned training process tree; shared pages can be counted more than once. **GPU MiB:** memory allocated by PyTorch in the training process at epoch end; CPU runs show unavailable. GPU utilization, temperature, power, and network traffic are not inferred.

Resource chart time is relative to each worker's own launch, avoiding cross-device wall-clock skew. Log timestamps are captured on the producing device; synchronize laptop clocks when inspecting cross-device log ordering. The logged round prefix is the last observed round at capture time.

## Export and retention

**CSV** exports round, client-round, epoch, and system measurements with a `record_type` column. **JSON** also includes configuration, checksums, class metrics/matrices, SCR support, and all captured process-log chunks. Both downloads require the logged-in operator and respect the service's run ownership.

Events are persisted in SQLite and forwarded by authenticated device heartbeats. Event IDs make repeated delivery idempotent. The monitor additionally normalizes repeated round/client/epoch callbacks by semantic identity, so they cannot inflate payload totals or round counts.

For responsiveness, charts show at most the latest 2,000 selected epoch/resource records and logs show the latest 250 chunks. Exports include all stored records, including runs exceeding 10,000 events. Filter a source to examine a longer client history. A source with no measurements stays empty.

Detailed stdout/stderr is relayed while a training task runs, including its exit traceback. Local task process logs remain available through **Open process log**. Older sessions may have round/epoch metrics but no confusion matrices, detailed diagnostics, or relayed logs. Missing measurements cannot be reconstructed from overall accuracy; run a new session to collect them.

## A useful SCR experiment sequence

1. Use the same dataset, partition identity, roster, model, seed, optimizer, epochs, and rounds for FedAvg and SCR. Keep practical/Dirichlet and physical-device/simulation results distinguishable.
2. Compare combined accuracy and per-class recall, then inspect the client/class combinations responsible for a change.
3. For SCR, inspect weighted risk and head penalty separately. An unusually large sample weight or update norm is a reason to investigate; it does not automatically mean a bug.
4. Compare fit and evaluation time alongside resource use. Parameter payload excludes monitoring overhead. Diagnostics add computation, storage, and heartbeat traffic; keep the same instrumentation enabled across methods for a fair runtime comparison.
5. Export both runs and repeat across seeds before deciding whether a change is useful. Do not tune on BTA and then present it as independent test evidence.

## Updating running services

After pulling these changes, rebuild the frontend as described in [TEAM_SETUP.md](TEAM_SETUP.md), finish/cancel any active run, and restart every coordinator/institution service. All training devices need the updated source; the readiness code-hash check will reject mixed training versions. Re-run device readiness checks before creating a new session.

From the repository directory, restart an existing instance using one of these commands. If both portals run on the same laptop, use a separate terminal for each:

```powershell
# Server/coordinator terminal
& .\.venv-web\Scripts\python.exe -m portal restart --state-dir .portal-coordinator
```

```powershell
# Client/institution terminal
& .\.venv-web\Scripts\python.exe -m portal restart --state-dir .portal-institution
```

The command refuses while a session or compute task is active. It stops only identified portal processes for that exact state directory, then serves the updated application in the current terminal. Keep that terminal open. Accounts, pairing, and results stay in the same state directory.
