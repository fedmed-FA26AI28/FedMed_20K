import os
import sys
import tkinter as tk
from pathlib import Path
from typing import Dict, Any, List

import customtkinter as ctk

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gui.components.env_banner import EnvBanner
from gui.components.server_tab import ServerTab
from gui.components.client_tab import ClientTab
from gui.components.plot_canvas import DualMetricPlotCanvas
from gui.components.log_console import LiveLogConsole
from gui.runners.process_runner import SubprocessRunner
from gui.runners.metric_parser import MetricStreamParser
from gui.telemetry import HardwareTelemetryMonitor, ping_server
from gui.config_manager import get_local_ip_addresses, check_port_in_use, kill_processes_on_port


class FedMedAIDashboard(ctk.CTk):
    """Main Application Window for FedMedAI Unified Federated Learning Dashboard."""

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.title("FedMedAI Federated Orchestrator & Live Monitor")
        self.geometry("1320x900")
        self.minsize(1160, 760)
        self.configure(fg_color="#0f172a")

        # Initialize Runners & Parsers
        self.metric_parser = MetricStreamParser(on_metric_callback=self._on_metric_event)
        self.server_runner = SubprocessRunner(
            on_log=self._on_server_log,
            on_exit=self._on_server_exit,
        )
        self.client_runner = SubprocessRunner(
            on_log=self._on_client_log,
            on_exit=self._on_client_exit,
        )

        self._build_layout()

        # Start Telemetry monitor
        self.telemetry = HardwareTelemetryMonitor(interval=2.5, callback=self._on_telemetry_update)
        self.telemetry.start()

        # Intercept window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self):
        # 1. Top Environment Bar (Full Width)
        self.banner = EnvBanner(self, on_venv_changed=self._on_venv_changed)
        self.banner.pack(fill="x", padx=12, pady=(10, 6))

        # 2. Main Tabview
        self.tabs = ctk.CTkTabview(
            self,
            fg_color="#131722",
            segmented_button_fg_color="#1e293b",
            segmented_button_selected_color="#2563eb",
            segmented_button_selected_hover_color="#1d4ed8",
            segmented_button_unselected_color="#1e293b",
            segmented_button_unselected_hover_color="#334155",
            text_color="#f1f5f9",
        )
        self.tabs.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        self.tab_server = self.tabs.add("🖥️ FL Server Control")
        self.tab_client = self.tabs.add("💻 FL Client Node")

        self._build_server_workspace()
        self._build_client_workspace()

    # --- SERVER WORKSPACE ---

    def _build_server_workspace(self):
        # 1. Top IP Ribbon across FULL WIDTH of the server tab
        ribbon = ctk.CTkFrame(
            self.tab_server,
            fg_color="#451a03",
            height=44,
            corner_radius=8,
            border_color="#f59e0b",
            border_width=1,
        )
        ribbon.pack(fill="x", padx=10, pady=(6, 8))
        ribbon.pack_propagate(False)

        lbl = ctk.CTkLabel(
            ribbon,
            text="📡 Server Host IP:",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#fde68a",
        )
        lbl.pack(side="left", padx=(14, 6))

        self.server_ip_list = get_local_ip_addresses()
        ip_options = [f"{label} ({ip})" for label, ip in self.server_ip_list]
        self.cmb_server_ip = ctk.CTkComboBox(
            ribbon,
            values=ip_options if ip_options else ["127.0.0.1"],
            width=240,
            height=28,
            font=ctk.CTkFont(size=11),
        )
        self.cmb_server_ip.pack(side="left", padx=4)

        lbl_port = ctk.CTkLabel(ribbon, text="Port:", font=ctk.CTkFont(size=11, weight="bold"), text_color="#fde68a")
        lbl_port.pack(side="left", padx=(10, 2))

        self.entry_server_port = ctk.CTkEntry(ribbon, width=65, height=28, font=ctk.CTkFont(family="monospace", size=11))
        self.entry_server_port.insert(0, "8080")
        self.entry_server_port.pack(side="left", padx=4)

        btn_copy = ctk.CTkButton(
            ribbon,
            text="📋 Copy IP:Port",
            width=115,
            height=28,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#d97706",
            hover_color="#b45309",
            command=self._copy_server_address,
        )
        btn_copy.pack(side="left", padx=10)

        self.lbl_server_port_status = ctk.CTkLabel(
            ribbon,
            text="● Port 8080 Ready",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#34d399",
        )
        self.lbl_server_port_status.pack(side="left", padx=10)

        btn_clean_port = ctk.CTkButton(
            ribbon,
            text="Clean Port",
            width=85,
            height=26,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#7f1d1d",
            hover_color="#991b1b",
            command=self._clean_server_port,
        )
        btn_clean_port.pack(side="right", padx=12)

        # 2. Main Two-Column Body below the ribbon
        container = ctk.CTkFrame(self.tab_server, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=4, pady=0)

        # Left Column: Configuration Form (width 400)
        self.server_ctrl = ServerTab(
            container,
            width=400,
            on_start=self._start_server,
            on_stop=self._stop_server,
            on_export=self._export_server_results,
        )
        self.server_ctrl.pack(side="left", fill="both", padx=(0, 6), pady=0)

        # Right Column: Visualizer (Top) + Log Console (Bottom)
        right_col = ctk.CTkFrame(container, fg_color="transparent")
        right_col.pack(side="right", fill="both", expand=True, padx=(6, 0), pady=0)

        self.server_plot = DualMetricPlotCanvas(right_col, mode="server")
        self.server_plot.pack(fill="both", expand=True, pady=(0, 6))

        self.server_console = LiveLogConsole(right_col, title="SERVER LIVE LOG CONSOLE")
        self.server_console.pack(fill="both", expand=True, pady=(6, 0))

    def _copy_server_address(self):
        val = self.cmb_server_ip.get()
        ip = val.split("(")[-1].replace(")", "").strip() if "(" in val else val.strip()
        port = self.entry_server_port.get().strip() or "8080"
        addr = f"{ip}:{port}"
        self.clipboard_clear()
        self.clipboard_append(addr)
        self.lbl_server_port_status.configure(text=f"Copied: {addr}", text_color="#67e8f9")

    def _clean_server_port(self):
        try:
            port = int(self.entry_server_port.get().strip())
            killed = kill_processes_on_port(port)
            if killed:
                self.lbl_server_port_status.configure(text=f"Killed PIDs: {killed}", text_color="#34d399")
            else:
                self.lbl_server_port_status.configure(text="Port is free", text_color="#34d399")
        except Exception as e:
            self.lbl_server_port_status.configure(text=f"Error: {e}", text_color="#f87171")

    # --- CLIENT WORKSPACE ---

    def _build_client_workspace(self):
        # 1. Top Target Connection Ribbon across FULL WIDTH of the client tab
        ribbon = ctk.CTkFrame(
            self.tab_client,
            fg_color="#1e293b",
            height=44,
            corner_radius=8,
            border_color="#3b82f6",
            border_width=1,
        )
        ribbon.pack(fill="x", padx=10, pady=(6, 8))
        ribbon.pack_propagate(False)

        lbl = ctk.CTkLabel(
            ribbon,
            text="🔗 Target FL Server Address:",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#93c5fd",
        )
        lbl.pack(side="left", padx=(14, 6))

        self.entry_client_server_addr = ctk.CTkEntry(
            ribbon,
            width=240,
            height=28,
            font=ctk.CTkFont(family="monospace", size=11),
        )
        self.entry_client_server_addr.insert(0, "127.0.0.1:8080")
        self.entry_client_server_addr.pack(side="left", padx=4)

        btn_ping = ctk.CTkButton(
            ribbon,
            text="⚡ Ping Server",
            width=110,
            height=28,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#2563eb",
            hover_color="#1d4ed8",
            command=self._ping_client_server,
        )
        btn_ping.pack(side="left", padx=10)

        self.lbl_client_ping_status = ctk.CTkLabel(
            ribbon,
            text="● Not checked",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#94a3b8",
        )
        self.lbl_client_ping_status.pack(side="left", padx=10)

        # 2. Main Two-Column Body below the ribbon
        container = ctk.CTkFrame(self.tab_client, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=4, pady=0)

        # Left Column: Client Parameters & Partition Inspector (width 400)
        self.client_ctrl = ClientTab(
            container,
            width=400,
            on_start=self._start_client,
            on_stop=self._stop_client,
            on_clear=self._clear_client_plot,
            get_python_exe=lambda: self.banner.active_venv,
        )
        self.client_ctrl.pack(side="left", fill="both", padx=(0, 6), pady=0)

        # Right Column: Client Visualizer (Top) + Client Console (Bottom)
        right_col = ctk.CTkFrame(container, fg_color="transparent")
        right_col.pack(side="right", fill="both", expand=True, padx=(6, 0), pady=0)

        self.client_plot = DualMetricPlotCanvas(right_col, mode="client")
        self.client_plot.pack(fill="both", expand=True, pady=(0, 6))

        self.client_console = LiveLogConsole(right_col, title="CLIENT LIVE LOG CONSOLE")
        self.client_console.pack(fill="both", expand=True, pady=(6, 0))

    def _ping_client_server(self):
        addr = self.entry_client_server_addr.get().strip()
        host = "127.0.0.1"
        port = 8080
        if ":" in addr:
            parts = addr.split(":")
            host = parts[0].strip()
            try:
                port = int(parts[1].strip())
            except ValueError:
                pass
        else:
            host = addr

        res = ping_server(host, port)
        if res["success"]:
            self.lbl_client_ping_status.configure(text=f"● {res['message']}", text_color="#34d399")
        else:
            self.lbl_client_ping_status.configure(text=f"● {res['message']}", text_color="#f87171")

    # --- PROCESS EVENT HANDLERS ---

    def _start_server(self, partial_cmd: List[str]):
        python_exe = self.banner.active_venv
        port = self.entry_server_port.get().strip() or "8080"
        cmd = self.server_ctrl.build_command(python_exe, port=port)
        self.server_console.append_log(f"\n[SERVER LAUNCH] Invoking command:\n{' '.join(cmd)}\n\n")
        self.server_plot.clear()

        ok = self.server_runner.start(cmd)
        if ok:
            self.server_ctrl.set_running_state(True)
        else:
            self.server_console.append_log("[ERROR] Server process could not be launched.\n")

    def _stop_server(self):
        self.server_console.append_log("[SERVER KILL] Terminating server process tree...\n")
        self.server_runner.stop()
        self.server_ctrl.set_running_state(False)

    def _export_server_results(self):
        self.server_plot.export_plot_image()

    def _on_server_log(self, line: str):
        self.server_console.append_log(line)
        self.metric_parser.parse_line(line, source="server")

    def _on_server_exit(self, code: int):
        self.server_console.append_log(f"\n[SERVER TERMINATED] Process exited with code {code}.\n")
        self.after(0, lambda: self.server_ctrl.set_running_state(False))

    def _start_client(self, partial_cmd: List[str]):
        python_exe = self.banner.active_venv
        server_addr = self.entry_client_server_addr.get().strip() or "127.0.0.1:8080"
        cmd = self.client_ctrl.build_command(python_exe, server_address=server_addr)
        self.client_console.append_log(f"\n[CLIENT LAUNCH] Invoking command:\n{' '.join(cmd)}\n\n")
        self.client_plot.clear()

        ok = self.client_runner.start(cmd)
        if ok:
            self.client_ctrl.set_running_state(True)
        else:
            self.client_console.append_log("[ERROR] Client process could not be launched.\n")

    def _stop_client(self):
        self.client_console.append_log("[CLIENT KILL] Terminating client process tree...\n")
        self.client_runner.stop()
        self.client_ctrl.set_running_state(False)

    def _clear_client_plot(self):
        self.client_plot.clear()

    def _on_client_log(self, line: str):
        self.client_console.append_log(line)
        self.metric_parser.parse_line(line, source="client")

    def _on_client_exit(self, code: int):
        self.client_console.append_log(f"\n[CLIENT TERMINATED] Process exited with code {code}.\n")
        self.after(0, lambda: self.client_ctrl.set_running_state(False))

    def _on_metric_event(self, source: str, payload: Dict[str, Any]):
        """Dispatched from metric_parser when new metrics are extracted from stdout."""
        if source == "server":
            r = payload.get("round", 1)
            loss = payload.get("loss")
            bta = payload.get("bta")
            bmcta = payload.get("bmcta")
            ta = payload.get("ta")
            mcta = payload.get("mcta")
            self.after(0, lambda: self.server_plot.update_server_metrics(r, loss, bta, bmcta, ta, mcta))
        elif source == "client_epoch":
            epoch = payload.get("epoch", 1)
            loss = payload.get("train_loss")
            acc = payload.get("train_acc")
            self.after(0, lambda: self.client_plot.update_client_metrics(epoch, train_loss=loss, train_acc=acc))
        elif source == "client_eval":
            ta = payload.get("ta")
            bta = payload.get("bta")
            r = self.client_plot.rounds[-1] if self.client_plot.rounds else 1
            self.after(0, lambda: self.client_plot.update_client_metrics(r, ta=ta, bta=bta))

    def _on_telemetry_update(self, info: Dict[str, Any]):
        self.server_console.update_telemetry(info)
        self.client_console.update_telemetry(info)

    def _on_venv_changed(self, new_venv: str):
        self.server_console.append_log(f"[CONFIG] Active interpreter updated to: {new_venv}\n")
        self.client_console.append_log(f"[CONFIG] Active interpreter updated to: {new_venv}\n")

    def _on_close(self):
        """Cleanup processes and stop telemetry before quitting."""
        self.telemetry.stop()
        self.server_runner.stop()
        self.client_runner.stop()
        self.destroy()


def main():
    app = FedMedAIDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
