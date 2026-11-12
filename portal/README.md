# FedMedAI LAN portal

**Setting up a teammate after a Git pull?** Follow the [detailed team setup guide](TEAM_SETUP.md), covering fresh environments, server/client roles, LAN ports, pairing, the first training run, prediction, and later updates.

Two real workspaces wrap the existing Flower server and `client.client`: a coordinator on the server laptop and an institution service on each client. Training uses the existing algorithms and partition functions. Browser requests, training, and prediction are separate workloads.

The English UI includes device pairing, environment checks, public dataset preparation, explicit participation, live training, recorded experiments, verified model manifests, and local image prediction. Assets and Inter fonts are bundled for offline LAN use.

The **Training → Experiment monitor** provides learning curves, SCR objective diagnostics, per-client comparisons, confusion matrices and class metrics, resource/timing charts, searchable process logs, and complete CSV/JSON exports. See [monitoring guide and metric definitions](MONITORING.md) for interpretation and a fair SCR/FedAvg experiment workflow.

## Setup on Windows

Use **Node.js 22.12 or newer** to build the frontend, a **Python 3.11+ web environment**, and a separate working training environment with the repository's training dependencies. The implementation was verified locally with Node 24, Python 3.14, Flower 1.37, and PyTorch 2.14. Jetson versions require their own compatibility check.

From the repository directory:

```powershell
# Builds web/dist and creates .venv-web. Does not replace your training environment.
.\scripts\setup_portal.ps1 -Python C:\Path\To\Python\python.exe

# On the server laptop: use its actual LAN IPv4 address, not localhost.
.\scripts\start_portal.ps1 -Mode coordinator -Name "FedMedAI Research Network" `
  -TrainerPython C:\Path\To\TrainingEnv\Scripts\python.exe -ServerAddress 192.168.1.10

# On each client laptop: give each institution a distinct name.
.\scripts\start_portal.ps1 -Mode institution -Name "Institution A" `
  -TrainerPython C:\Path\To\TrainingEnv\Scripts\python.exe
```

First startup prompts for an operator password; the username is `admin`. Open `http://SERVER_IP:8000` for the coordinator and `http://CLIENT_IP:8001` for each institution. To use another username, initialize directly with `python -m portal init --username ...`.

The launcher creates configuration only on first use. Subsequent launches reuse `<state-dir>/settings.json`. Stop the service before editing its address, ports, interpreter, dataset root, or name. Each device needs its own state directory; never copy a live `.portal` directory to another device.

Allow inbound TCP **8000 and 8080 on the coordinator**, and **8001 on institutions** on the trusted private LAN. Client agents call the coordinator API; Flower clients connect to its Flower port. Pairing does not configure Windows Firewall or the router. Guest Wi-Fi with client isolation cannot support this demo.

## First complete workflow

1. Sign in on every service and check **Overview → Device readiness**. A separate worker checks actual Python, Torch, Flower, CUDA, and code identity.
2. On institutions, open **Dataset → Prepare dataset**. On the coordinator, use **Overview → Benchmark datasets → Prepare dataset**. The first request may download public MedMNIST files. All devices must have the same public benchmark file.
3. On the coordinator, choose **Institutions → Pair an institution**. Enter its URL and one-time token on the client **Overview → Connect to coordinator**. Create a different token for each client.
4. Create a session with the selected client roster. The coordinator generates and hashes an unchanged research partition for exactly that roster.
5. On each institution, accept the invitation. Readiness checks compare source code, Flower version, dataset checksum, partition checksum, assigned indices, and training class counts.
6. Start training from the coordinator after every client is ready. Track actual rounds and client states. Closing a browser does not stop the services.
7. Completed sessions register final checkpoints. Institution **Models** can download global models from the coordinator. FedBN/FedBNS models remain specific to their owning institution.
8. Use **Predict** on an institution, choose a verified model, then select a local sample or upload a PNG/JPEG benchmark-style patch.

The default session is BloodMNIST, SCR, baseline_bn, Dirichlet 0.3, seed 42, three rounds, one local epoch, batch 16, SGD 0.01, momentum 0.9, weight decay 0, no augmentation, local test evaluation. Five live methods are supported: FedAvg, FedBN, SCAFFOLD, FedBNS, SCR.

### Prepare data without opening the institution UI

Run in the **training** environment. `root` must equal the portal's `data_root` (default: the current user's `.medmnist` directory).

```powershell
python -c "import medmnist; from pathlib import Path; root=Path.home()/'.medmnist'; root.mkdir(exist_ok=True); [getattr(medmnist, medmnist.INFO[d]['python_class'])(split='train', root=str(root), download=True) for d in ['bloodmnist','organamnist','pathmnist']]"
```

Training and prediction never download automatically. For a Nano, prepare the files before the defense, then use the same data root for its training environment and portal.

## Jetson compatibility gate

Build `web/dist` once on a laptop and copy it alongside the Python code. No Node.js build process is needed on a Nano. Create a separate Python 3.11+ environment for the web service and install `portal/requirements.txt`; keep the existing NVIDIA-compatible training Python separate.

```bash
export FEDMED_WEB_PYTHON=/path/to/web-env/bin/python
export FEDMED_TRAINER_PYTHON=/path/to/training-env/bin/python
export FEDMED_NAME="Jetson institution 1"
bash scripts/start_portal.sh institution
```

Preflight reports the actual platform, Jetson release string, Python, Torch, CUDA, and Flower versions. A successful import check is necessary but not proof that training works: complete a real one-round rehearsal on each Nano. Devices must run the same training code and Flower version. If an older Nano cannot support that runtime, resolve its environment separately before including it in the roster. These scripts do not flash firmware or replace JetPack/CUDA.

## Results, model identity, and failure behavior

- **Recorded experiments:** import from the configured artifact roots (by default `results` in this repository and `../results`). Duplicate round records are merged without modifying the original files. The UI shows distinct rounds and explicit BTA/BMCTA/final metrics. Missing partition hashes remain unknown.
- **Model import:** use project-generated `.pth` state dictionaries under an allowed artifact directory. Supply the original dataset, architecture, method, run, round, and selection rule. Workers load with `weights_only=True`, require a strict state-dictionary match, and verify finite model output. No arbitrary uploaded Python/model code runs.
- **Model scope:** global inference models support FedAvg/SCAFFOLD/SCR. FedBN/FedBNS checkpoints are saved after local evaluation, preserving the institution's BatchNorm state. A server checkpoint alone is insufficient for these methods.
- **Privacy boundary:** public MedMNIST data is preloaded on devices. During training, Flower exchanges model parameters and metrics; SCR additionally uses training class-count metadata. Prediction images go to the selected institution service and are removed from worker request storage on completion. This is a LAN research demo, with no claim of encrypted Flower transport, secure aggregation, or clinical validation.
- **Compute:** each service runs one compute task at a time. Prediction is unavailable while that institution is training or preparing data. Null telemetry means unavailable, never an invented zero.
- **Failure:** missing clients cannot silently change the roster. Round operations have a timeout. Cancellation terminates only a recorded, identity-checked process tree. Restarts mark unfinished sessions interrupted; they never pretend to resume optimizer or SCAFFOLD state.

## Development and verification

```powershell
python -m portal export-openapi --output web/openapi.json
npx pnpm@11.25.0 --dir web generate:api
npx pnpm@11.25.0 --dir web build
python -m unittest portal.tests.test_portal portal.tests.test_workers -v

# Uses real cached BloodMNIST, three local client subprocesses, one round per method.
# Writes only under ignored .portal-qa. Does not replace benchmark results.
python -m portal.tests.smoke --cpu
```

For frontend development, run `npx pnpm@11.25.0 --dir web dev` with the coordinator API on port 8000. The Vite development proxy forwards `/api`. The deployed build is served by FastAPI, so CORS configuration is unnecessary.

Operator writes require the `X-FedMed-Request: 1` header and same-origin browser requests. Cookies are HttpOnly/SameSite Strict. Devices authenticate separately using paired bearer tokens. Never expose this first release directly to the public internet.

See [Vietnamese demo guide](DEMO_VI.md) for a presentation sequence and the physical-device acceptance checklist.
See [verification record](VERIFICATION.md) for completed checks, local evidence, and the remaining hardware gate.

The interface follows the [React application setup guidance](https://react.dev/learn/build-a-react-app-from-scratch). Compute workers run outside HTTP request handling, consistent with the distinction between small background work and heavy computation in [FastAPI's background-task guidance](https://fastapi.tiangolo.com/tutorial/background-tasks/).
