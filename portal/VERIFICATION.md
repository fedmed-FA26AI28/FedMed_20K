# Verification record — 12 November 2026

This release is a local/LAN research demonstration. The checks below do not establish a medical accuracy improvement or complete the physical-device acceptance test.

## Completed checks

| Check | Evidence / result |
|---|---|
| Real Flower training | FedAvg, FedBN, SCAFFOLD, FedBNS, and SCR each completed one round with three separate `client.client` processes on cached public BloodMNIST. Existing Dirichlet 0.3 partition algorithm, seed 42, baseline_bn, one epoch, batch 32. |
| Exact short demo preset | SCR also completed three rounds, one epoch per round, batch 16, with three CPU clients. Client round counts and stopped elapsed timers were verified, along with final checkpoint registration and prediction parity. |
| Prediction parity | Each trained method's checkpoint produced matching class and scores through the institution prediction API and direct Python inference. Independent reference preprocessing/inference tests cover RGB BloodMNIST, grayscale OrganAMNIST, and RGB PathMNIST. |
| Local BatchNorm | FedBN/FedBNS checkpoints contain learned local BN buffers. A focused empty-test check confirms incoming global state does not overwrite those buffers, even when evaluation has zero images. |
| Existing research checkpoints | Real saved baseline_bn checkpoints for all three datasets were imported, downloaded to the institution, checked against their checksums/architectures, and used for local upload inference. Generated images in this check verify plumbing, not accuracy. |
| API / worker regression tests | 15 tests passed: authentication/origin checks, role restrictions, pairing-token reuse, event ownership/deduplication, process cancellation/PID identity, restart interruption, failed roster, unavailable Python, missing/foreign/incompatible/changed checkpoints, empty tests, prediction transforms, strict roster rejection, and duplicate communication accounting. |
| Browser workflow | Headless Chrome against real local services: login, pairing, session form, real run history, read-only experiment import, checkpoint import, page refresh, assigned dataset, real prediction, and responsive layout. No browser runtime errors or external asset/API requests. |
| Frontend | Strict TypeScript check and production Vite build passed; fonts, icons, and assets are bundled locally. |

The local runtime was Windows 11, Python 3.14.7, PyTorch 2.14.0+cu126, Flower 1.37.0, MedMNIST 3.0.2, and Node 24.22.1. This is evidence for this runtime, not a version guarantee for Jetson.

FedAvg and FedBN passed with CUDA. SCAFFOLD passed in a run whose attempted CPU override did not disable CUDA on Windows. The harness now uses `CUDA_VISIBLE_DEVICES=-1` and asserts every client reports CUDA disabled for `--cpu`; FedBNS and SCR passed with that verified CPU setting. Earlier simultaneous GPU attempts encountered CUDA unknown-error/out-of-memory failures. The app reported these as failed sessions. A physical client normally has its own compute device, but it must still be rehearsed.

## Local artifacts

Test state and screenshots are ignored by Git and kept separate from research result folders:

- `.portal-qa/smoke-20261112-003539/report.json`: FedAvg.
- `.portal-qa/smoke-20261112-003846/report.json`: FedBN.
- `.portal-qa/smoke-20261112-004536/report.json`: SCAFFOLD (actual CUDA, see correction above).
- `.portal-qa/smoke-20261112-005012/report.json`: FedBNS/SCR with CPU explicitly verified.
- `.portal-qa/smoke-20261112-005949/report.json`: three-round SCR demonstration preset with CPU explicitly verified.
- `.portal-qa/import-report.json`: saved-checkpoint imports for all three datasets.
- `web/test-results/browser/report.json` and PNGs: browser checks and screenshots.

The smoke-test instances use test-only accounts and loopback addresses. They are not deployment configuration. The regular launcher creates a new instance with an operator-selected password.

## Detailed experiment monitoring update

The monitoring update was checked locally on November 12, 2026:

- 22 tests passed in `python -m unittest portal.tests.test_monitor portal.tests.test_portal portal.tests.test_workers -q`, followed by an additional passing isolated-process restart test. The resulting suite contains 23 tests. Coverage includes unchanged FedAvg/SCR model updates with telemetry on/off, directly computed confusion counts, macro/weighted metric semantics, skipped empty tests, semantic event deduplication, exports past 10,000 events, run ownership, log relaying, non-finite diagnostic values, and restart refusal during active work.
- Real one-round CPU runs with three Flower client processes completed for **FedAvg, FedBN, SCAFFOLD, FedBNS, and SCR**. All produced epoch diagnostics, resource samples, process logs, confusion matrices, update norms, and parameter payload measurements. Checkpoint scope and direct-inference parity continued to pass.
- All five pooled confusion-matrix accuracies matched their completed-round accuracy within `1e-6`; each covered the 3,421 assigned BloodMNIST test images. This is integration evidence, not a comparative research result.
- TypeScript checking and the Vite production build passed. Browser checks against real recorded FedAvg and SCR runs covered charts, comparison selection, client filtering, confusion tables, log search, CSV/JSON downloads, pause, reload, and mobile overflow. Screenshots were inspected.
- The restart command refused the user's active OrganAMNIST session without stopping any process. An isolated positive test verified that it stops only the matching idle portal and preserves an unrelated process, including Windows console-host handling.

Evidence:

- `.portal-qa/smoke-20261112-131905/report.json`
- `.portal-qa/smoke-20261112-131905/monitor-validation.json`
- `web/test-results/monitor/fedavg-report.json`, `scr-report.json`, and screenshots.

These checks do not measure monitoring overhead in a controlled benchmark or constitute physical multi-laptop/Nano acceptance. Keep the same instrumentation enabled across compared methods. Older sessions cannot reconstruct diagnostics that were never recorded. See [MONITORING.md](MONITORING.md) for definitions and restart instructions.

## Still requiring physical hardware

- Server laptop plus three separate client laptops on the actual LAN.
- At least one Jetson Nano with compatible Python, JetPack, Torch/CUDA, Flower, memory, and network settings.
- Full demonstration preset (three rounds, batch 16) on that roster; measure actual runtime.
- Real cable/Wi-Fi interruption and firewall checks on that LAN.
- Offline rehearsal after provisioning dependencies, all required datasets, and models.

No Nano was accessed or modified. No full 80-round experiment was run. OrganAMNIST dataset training was not rerun; its class mapping, preprocessing, checkpoint loading, and uploaded-image prediction path were verified.
