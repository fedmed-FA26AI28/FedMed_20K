# Research experiment and decision log

Use dated entries to preserve evidence across app-debugging detours and new chats.
Record planned, running, completed, failed, and inconclusive work accurately.
See [the plan](RESEARCH_PLAN.md) and [the short handoff](../../CHAT_HANDOFF.md).

## 2026-11-13 — Historical Dirichlet 0.3 review

**Status:** completed read-only review; no new training was launched.

Sources: [results/simulation](../../results/simulation), specifically
`{dataset}_dirichlet_0.3_{strategy}_baseline_bn_*/{dataset}/dirichlet_0.3/{strategy}/fl_results.json`,
with the enclosing `simulation_results.json` and `terminal.log` used for context.
Use seeds in metadata: folder suffixes do not consistently identify seed order.

Each available group below has five seeds, 42–46, and 80 distinct rounds. BTA is
reported as mean +/- sample standard deviation, in percent. Summary BTA was
cross-checked against the maximum recorded WCTA trajectory for FedAvg, SCR, and
FedBN. These are historical results, not newly reproduced matched experiments.

| Dataset | FedAvg BTA | SCR BTA | SCR minus FedAvg, percentage points | FedBN BTA | FedBNS BTA |
|---|---:|---:|---:|---:|---:|
| BloodMNIST | 92.55 +/- 0.47 | 92.43 +/- 0.36 | -0.12 | 95.09 +/- 0.75 | Not found in the reviewed folder group |
| OrganAMNIST | 87.20 +/- 0.45 | 87.06 +/- 0.42 | -0.14 | 90.41 +/- 0.85 | 90.71 +/- 0.90 |
| PathMNIST | 83.73 +/- 1.16 | 83.79 +/- 0.87 | +0.05 | 82.90 +/- 2.95 | 84.10 +/- 2.45 |

Recorded setup: `baseline_bn`, 12 clients, 80 requested rounds, 5 local epochs,
batch size 256, SGD learning rate 0.01 fixed, momentum 0.9, weight decay 0,
no augmentation, local test evaluation, `client.client`. Check actual execution
logs before claiming every historical configuration field was effective.

**Limitations:** partition hashes were absent from the reviewed `fl_results.json`
metadata. The inspected SCR files contain 160 round records for 80 distinct round
IDs; duplicate records are not necessarily identical. Some aggregate/final fields
need reconciliation against phase-specific trajectories. Do not sum duplicate
communication/compute records or treat 160 as actual completed rounds. Runtime
comparisons also need matched hardware and execution conditions. BTA selects on
test data, and the small differences do not establish a dataset-size effect.

**Decisions:** retain original SCR; investigate absolute class support; separate
weighting, protection, and normalization in ablations. Start with BloodMNIST,
then confirm across all three datasets. Avoid a general improvement/SOTA claim.

**Next:** Step 1 of the research plan. The new absolute-count and directional
variants remain unimplemented and untested.

## 2026-11-13 — Continuity documents created

**Status:** research notes consolidated into the handoff, plan, and this log.
No training behavior, partition algorithm, accounts, or research results changed.
Source checkout at this point: `demo`, `5947d964e`. Recheck before subsequent work.

## Template for the next entry

```markdown
## YYYY-MM-DD — Experiment or decision name

- Status: planned / running / completed / failed / inconclusive
- Question and hypothesis:
- Branch, commit, and any uncommitted training changes:
- Dataset, split, partition/subset hash, client count, and seeds:
- Model, initialization, optimizer, epochs/steps, batch size, and method settings:
- Hardware, execution module, process settings, and instrumentation:
- Validation/tuning protocol and checkpoint selection rule:
- Artifact directory and exact launch command/notebook cell:
- Observed results, paired differences, uncertainty, and runtime:
- Failures, missing measurements, or comparability limits:
- Decision and reason:
- Next unfinished action:
```
