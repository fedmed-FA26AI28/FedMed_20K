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

## Still requiring physical hardware

- Server laptop plus three separate client laptops on the actual LAN.
- At least one Jetson Nano with compatible Python, JetPack, Torch/CUDA, Flower, memory, and network settings.
- Full demonstration preset (three rounds, batch 16) on that roster; measure actual runtime.
- Real cable/Wi-Fi interruption and firewall checks on that LAN.
- Offline rehearsal after provisioning dependencies, all required datasets, and models.

No Nano was accessed or modified. No full 80-round experiment was run. OrganAMNIST dataset training was not rerun; its class mapping, preprocessing, checkpoint loading, and uploaded-image prediction path were verified.
