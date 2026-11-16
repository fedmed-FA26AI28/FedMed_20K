# Research handoff: SCR and selective drift correction

Last updated: 2026-11-13 (Asia/Saigon).

Read this first when continuing the capstone in a new chat. This is the short
entry point; the detailed research decisions belong in the linked documents.

## Start a new chat

Copy this prompt:

> Read `CHAT_HANDOFF.md`, `docs/research/RESEARCH_PLAN.md`, and
> `docs/research/EXPERIMENT_LOG.md` in this repository. Check the current Git
> branch and working tree before acting. Continue from the first unfinished
> research step while preserving the recorded experimental constraints.
> Distinguish evidence from hypotheses. I run Kaggle jobs myself; provide
> notebook cells when a GPU experiment is needed.

## Research status

- **Project:** federated medical image classification on BloodMNIST,
  OrganAMNIST, and PathMNIST, initially motivated by FedSLD.
- **Existing method:** support-calibrated risk (SCR): fixed training-count
  weighting plus a support-dependent classifier anchor. It retains
  sample-weighted FedAvg aggregation and raw-logit prediction.
- **Current question:** can correction based on absolute class support preserve
  global knowledge on clients with scarce or missing classes, at low cost?
- **Evidence:** the reviewed Dirichlet 0.3 SCR means are very close to FedAvg.
  PathMNIST's ranking does not establish that SCR benefits from larger datasets.
  FedBN's larger gains on BloodMNIST and OrganAMNIST make normalization an
  important independent factor. See the experiment log for numbers and caveats.
- **Status of new ideas:** proposals only. Absolute-count protection and
  directional correction have not been implemented or trained as part of this
  research discussion. No improvement, speedup, novelty, or SOTA claim is proven.

## Next unfinished step

**Step 1 in the research plan: audit reproducibility and evaluation readiness.**
Locate the intended implementation branch, confirm runnable entrypoints, recover
or regenerate partitions with the existing algorithm, record their hashes, and
define validation-selected evaluation. Then prepare the BloodMNIST ablations.
Do not skip directly to a large experiment matrix or combine all proposed ideas.

## Constraints to carry forward

- Lead experiment: `baseline_bn`, 12 clients, Dirichlet alpha 0.3. Keep practical,
  Dirichlet 0.1, Dirichlet 0.3, and physical-device experiments distinct.
- Reference settings: 80 rounds, 5 local epochs, batch size 256, SGD, learning
  rate 0.01 fixed, momentum 0.9, weight decay 0, no augmentation; use
  `client.client`. Existing SCR uses gamma 1 and head lambda 0.01.
- Preserve the partition algorithm and empty-local-test handling. Use training
  counts only for support calculations. Record deliberate setting changes.
- Keep original SCR available. Give every candidate an explicit, separate
  configuration. Keep sample-weighted aggregation and raw-logit inference unless
  an experiment explicitly studies a change to them.
- Use validation for tuning and checkpoint selection. BTA and BMCTA are
  exploratory best-test-round statistics, not unbiased final-test estimates.
- Compare seeds by their stored metadata, not folder suffixes. Match hardware,
  process settings, and evaluation work when comparing runtime.
- The user runs Kaggle personally. Supply reviewable notebook cells; avoid the
  older `test.sh`, which automatically commits and pushes results.
- Retaining images locally does not establish differential privacy or secure
  aggregation. Do not describe SCR or the candidate as formally private.

## Checkout snapshot, not a permanent branch assumption

On 2026-11-13, `D:\FPT\KLTN\Code\FedMed_20K` was on `demo` at `5947d964e`.
`algorithms/scr.py` and the reviewed results existed. These previously mentioned
files were absent in that checkout: `algorithms/SCR.md`,
`scripts/run_scr_dirichlet01.py`, `algorithms/fedsol.py`, and
`algorithms/fedswitch.py`. Check other branches/worktrees before assuming the
files were lost. Do not switch or merge branches merely to match this snapshot.

## Keep continuity during bug-fixing conversations

Use [the research plan](docs/research/RESEARCH_PLAN.md) for hypotheses and ordered
steps, and [the experiment log](docs/research/EXPERIMENT_LOG.md) for dated evidence
and decisions. Record a bug fix there only when it affects training, evaluation,
timing, or reproducibility. Portal/LAN setup belongs in
[TEAM_SETUP.md](portal/TEAM_SETUP.md).

After research work, update the completed steps, actual findings, and the next
unfinished step here. Keep credentials and machine-specific portal state out of
these documents. A detour into app debugging does not replace the research goal.
