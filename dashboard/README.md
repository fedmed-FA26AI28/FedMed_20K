# FedMedAI dashboard

Tài liệu tiếng Việt: [Hướng dẫn dashboard và triển khai server/client](INSTRUCTION_DASHBOARD.md).

Run the dashboard on the central server only. Remote clients run the standard
Flower training client in a terminal; NiceGUI is not required on client machines.

```bash
python -m pip install -r requirements-dashboard.txt
python app_nicegui.py
```

Open http://localhost:8080. The default host is `0.0.0.0` for LAN access.
Use `--host 127.0.0.1` for local access, `--port 8088` to change the web port,
and `--no-browser` to skip opening a browser. Python 3.10+ is required; install
training dependencies from `requirements.txt` in the same Python environment.
`run_nicegui.py` uses the same launcher with defaults `127.0.0.1:8088`.
The desktop launcher `run_dashboard.py` is independent.

Web and Flower gRPC ports must differ: defaults are web **8080**, gRPC **8081**.
If gRPC is busy, choose another port or stop the process using it.

## Workflow

1. **FL Server**: choose dataset, Baseline/BaselineBN model, strategy, partition,
   training settings and client count. Require
   `Min Fit <= Min Available <= Total clients`. Start a LAN server or local
   simulation. Simulation runs separately and uses every client each round.
2. **Client Nodes** monitors clients reporting to the selected server run. The
   compact **Client configuration** form accepts the server IP:port and client ID,
   downloads `client_dashboard.yaml`, and copies the CLI command. It uses the
   selected LAN server run's configuration, or the server form when no run is selected.
   Save the YAML as `configs/client_dashboard.yaml` on the remote device.
   table shows IP, device, round, report age, CPU/RAM, VRAM, ping, and metrics.
   Reports arrive at the end of rounds; this is not an online heartbeat.
   Summary cards show configured/observed clients, fit reports in the current
   round, and clients still awaiting a fit report. Inspect client shows sample
   counts, epochs, learning rate, fit/epoch timings, payload size, per-client
   train/test curves and round history with CSV export. Fields require updated
   client code; missing reports remain blank. Full remote terminal logs are not
   collected by the server dashboard.
3. On each remote PC/Jetson, install matching project code and data. Use
   **Export server YAML** on the server, copy the file to the client as
   `configs/client_dashboard.yaml`, then run:
   ```bash
   python -m pip install -r requirements-client.txt
   python -m client.client --config configs/client_dashboard.yaml --client_id 0 --server_address 192.168.1.2:8081
   ```
   Replace the address and assign a unique ID from 0 to total clients minus one.
   PyTorch on Jetson must match the installed JetPack version. Client machines
   need neither NiceGUI nor a dashboard process. Stop a client with Ctrl+C in its
   terminal. Detailed setup instructions are maintained in this document.
4. **Live Analytics**: train/test accuracy, BTA, MCTA, BMCTA, loss and round
   metrics. BMCTA is best mean client test accuracy. Missing values stay blank.
5. **Live Terminal**: filter logs by run, level and keyword; clear the current
   browser view or copy its visible lines. Logs are bounded in memory.
6. **Cluster & Data**: host CPU/RAM/GPU telemetry and Non-IID class counts from
   existing `data/partitions/*.json`. This view does not download datasets.

Use **Export server YAML** to download the configuration. Each run also saves
`config.yaml`, `console.log` and `summary.json` in `results/dashboard/<run-id>/`.
Server/simulation training outputs are under its `artifacts/` directory.
Saved CSV results can be read directly with your usual analysis tools.

## Lifecycle and telemetry

Structured stdout events are enabled only for dashboard jobs. Round reports
from Flower clients provide remote node telemetry. Host GPU sensors use NVIDIA
SMI or Jetson sysfs; CUDA detection falls back to PyTorch. Client VRAM is memory
allocated by that process; host VRAM is device-wide. Missing sensors show a dash.

Closing a browser does not stop jobs. Shutting down the dashboard stops managed
processes and their children. Stop requests Ctrl+C/CTRL_BREAK before forced
termination after a timeout. Jobs are not reattached after a dashboard restart;
their saved files remain available. This console is intended for a trusted LAN
and has no authentication.

## Verification

```bash
python -m unittest discover -s tests -p "test_dashboard*.py" -v
```

Tests exercise configuration, metric events, partitions, TCP probes, concurrent
processes, duplicate client IDs, stop/descendant cleanup and saved run files.
They do not train on MedMNIST.
