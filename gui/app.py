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

from gui.components.server_tab import ServerTab
from gui.components.client_tab import ClientTab
from gui.components.plot_canvas import DualMetricPlotCanvas
from gui.components.log_console import LiveLogConsole
from gui.runners.process_runner import SubprocessRunner
from gui.runners.metric_parser import MetricStreamParser
from gui.telemetry import HardwareTelemetryMonitor, ping_server
from gui.config_manager import get_local_ip_addresses, check_port_in_use, kill_processes_on_port, get_default_venv_path
from gui.components.env_banner import VenvDiagnosticModal


class FLOrchestratorDashboard(ctk.CTk):
    """Main Application Window for Unified Federated Learning Orchestrator & Live Monitor."""

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("Light")

        self.title("FL Orchestrator & Live Monitor")
        self.geometry("1400x900")
        self.minsize(1180, 760)
        self.configure(fg_color="#f5f7fa")

        # Active Virtual Environment Interpreter
        self.active_venv = get_default_venv_path()

        # Active Navigation View: "server" or "client"
        self.active_view = "server"

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
        # 1. FIXED LEFT SIDEBAR (Width: 230px, Deep Navy: #102135)
        self.sidebar = ctk.CTkFrame(self, width=230, corner_radius=0, fg_color="#102135")
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Brand Header
        brand_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand_frame.pack(fill="x", padx=16, pady=(20, 14))

        brand_top = ctk.CTkFrame(brand_frame, fg_color="transparent")
        brand_top.pack(fill="x")

        lbl_icon = ctk.CTkLabel(
            brand_top,
            text="⚕",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#63c7bb",
            width=28,
        )
        lbl_icon.pack(side="left", padx=(0, 4))

        lbl_brand_fl = ctk.CTkLabel(
            brand_top,
            text="FL",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#ffffff",
        )
        lbl_brand_fl.pack(side="left")

        lbl_brand_orch = ctk.CTkLabel(
            brand_top,
            text="Orchestrator",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#63c7bb",
        )
        lbl_brand_orch.pack(side="left", padx=(2, 0))

        lbl_subtitle = ctk.CTkLabel(
            brand_frame,
            text="MEDICAL RESEARCH PLATFORM",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color="#617890",
        )
        lbl_subtitle.pack(anchor="w", padx=(32, 0), pady=(2, 0))

        # Navigation Label
        lbl_nav = ctk.CTkLabel(
            self.sidebar,
            text="NAVIGATION",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color="#617890",
        )
        lbl_nav.pack(anchor="w", padx=20, pady=(16, 6))

        # Navigation Buttons
        self.btn_nav_server = ctk.CTkButton(
            self.sidebar,
            text="  🖥️  FL Server Control",
            height=38,
            anchor="w",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#1a3c48",
            hover_color="#1b3149",
            text_color="#a7e6d9",
            command=lambda: self._switch_view("server"),
        )
        self.btn_nav_server.pack(fill="x", padx=12, pady=3)

        self.btn_nav_client = ctk.CTkButton(
            self.sidebar,
            text="  📱  FL Client Node",
            height=38,
            anchor="w",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="transparent",
            hover_color="#1b3149",
            text_color="#96aabe",
            command=lambda: self._switch_view("client"),
        )
        self.btn_nav_client.pack(fill="x", padx=12, pady=3)

        # Bottom Sidebar Box
        bottom_box = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        bottom_box.pack(side="bottom", fill="x", padx=14, pady=16)

        sep = ctk.CTkFrame(bottom_box, height=1, fg_color="#25394e")
        sep.pack(fill="x", pady=(0, 10))

        status_row = ctk.CTkFrame(bottom_box, fg_color="transparent")
        status_row.pack(fill="x")

        lbl_dot = ctk.CTkLabel(status_row, text="●", font=ctk.CTkFont(size=12, weight="bold"), text_color="#39ad8a", width=16)
        lbl_dot.pack(side="left")

        self.lbl_sidebar_cluster = ctk.CTkLabel(
            status_row,
            text="Flower 1.37 · PyTorch 2.14",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#c7d4e0",
        )
        self.lbl_sidebar_cluster.pack(side="left", padx=4)

        self.lbl_sidebar_telemetry = ctk.CTkLabel(
            bottom_box,
            text="CPU: Scanning... | RAM: ... GB",
            font=ctk.CTkFont(family="monospace", size=9),
            text_color="#71879e",
        )
        self.lbl_sidebar_telemetry.pack(anchor="w", padx=20, pady=(4, 8))

        # Change/Test Venv Buttons
        venv_btns = ctk.CTkFrame(bottom_box, fg_color="transparent")
        venv_btns.pack(fill="x", pady=(2, 0))

        btn_ch_venv = ctk.CTkButton(
            venv_btns,
            text="Change Venv",
            height=26,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#1c334d",
            hover_color="#223f5e",
            text_color="#c7d4e0",
            command=self._change_venv,
        )
        btn_ch_venv.pack(side="left", fill="x", expand=True, padx=(0, 3))

        btn_t_venv = ctk.CTkButton(
            venv_btns,
            text="Test",
            height=26,
            width=50,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#0c8c80",
            hover_color="#087b70",
            text_color="#ffffff",
            command=self._test_venv,
        )
        btn_t_venv.pack(side="right", padx=(3, 0))

        # 2. MAIN WORKSPACE CONTAINER (Right side on #f5f7fa)
        self.main_container = ctk.CTkFrame(self, fg_color="#f5f7fa", corner_radius=0)
        self.main_container.pack(side="right", fill="both", expand=True)

        # 2A. TOPBAR (White #ffffff, height 54px, border bottom #e8edf2)
        self.topbar = ctk.CTkFrame(self.main_container, height=54, corner_radius=0, fg_color="#ffffff")
        self.topbar.pack(fill="x")
        self.topbar.pack_propagate(False)

        topbar_border = ctk.CTkFrame(self.main_container, height=1, corner_radius=0, fg_color="#e8edf2")
        topbar_border.pack(fill="x")

        # Topbar Left: Breadcrumb
        self.lbl_breadcrumb = ctk.CTkLabel(
            self.topbar,
            text="FL Platform  ›  FL Server  ›  FL Server Control & Live Monitor",
            font=ctk.CTkFont(size=11),
            text_color="#4c6176",
        )
        self.lbl_breadcrumb.pack(side="left", padx=20)

        # Topbar Right: Dynamic Host / Target Ribbon
        self.ribbon_frame = ctk.CTkFrame(self.topbar, fg_color="transparent")
        self.ribbon_frame.pack(side="right", padx=16)

        self._build_server_topbar_ribbon()
        self._build_client_topbar_ribbon()

        # 2B. 4-CARD KPI METRICS ROW
        self._build_kpi_row()

        # 2C. WORKSPACES (Server Workspace vs Client Workspace)
        self.workspace_body = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.workspace_body.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        self._build_server_workspace()
        self._build_client_workspace()

        # Bind reactive listeners to input fields for instant KPI updates
        self._bind_kpi_reactive_inputs()

        # Initial view
        self._switch_view("server")

    # --- TOPBAR RIBBONS ---

    def _build_server_topbar_ribbon(self):
        self.server_ribbon = ctk.CTkFrame(self.ribbon_frame, fg_color="transparent")

        lbl = ctk.CTkLabel(self.server_ribbon, text="Host IP:", font=ctk.CTkFont(size=11, weight="bold"), text_color="#78889b")
        lbl.pack(side="left", padx=(4, 4))

        self.server_ip_list = get_local_ip_addresses()
        ip_options = [f"{label} ({ip})" for label, ip in self.server_ip_list]
        self.cmb_server_ip = ctk.CTkComboBox(
            self.server_ribbon,
            values=ip_options if ip_options else ["127.0.0.1"],
            width=210,
            height=28,
            font=ctk.CTkFont(size=11),
            fg_color="#ffffff",
            border_color="#dce3eb",
            text_color="#162d43",
        )
        self.cmb_server_ip.pack(side="left", padx=4)

        lbl_port = ctk.CTkLabel(self.server_ribbon, text="Port:", font=ctk.CTkFont(size=11, weight="bold"), text_color="#78889b")
        lbl_port.pack(side="left", padx=(6, 2))

        self.entry_server_port = ctk.CTkEntry(
            self.server_ribbon,
            width=60,
            height=28,
            font=ctk.CTkFont(family="monospace", size=11),
            fg_color="#ffffff",
            border_color="#dce3eb",
            text_color="#162d43",
        )
        self.entry_server_port.insert(0, "8080")
        self.entry_server_port.pack(side="left", padx=4)

        btn_copy = ctk.CTkButton(
            self.server_ribbon,
            text="📋 Copy",
            width=70,
            height=28,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#ffffff",
            hover_color="#f3f7fa",
            text_color="#42566b",
            border_width=1,
            border_color="#dfe6ed",
            command=self._copy_server_address,
        )
        btn_copy.pack(side="left", padx=6)

        self.lbl_server_port_status = ctk.CTkLabel(
            self.server_ribbon,
            text="● Port 8080 Ready",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#0b8a5a",
        )
        self.lbl_server_port_status.pack(side="left", padx=6)

        btn_clean_port = ctk.CTkButton(
            self.server_ribbon,
            text="Clean Port",
            width=75,
            height=26,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#ffffff",
            hover_color="#fdf2f2",
            text_color="#ba4c4c",
            border_width=1,
            border_color="#fecaca",
            command=self._clean_server_port,
        )
        btn_clean_port.pack(side="left", padx=(6, 0))

    def _build_client_topbar_ribbon(self):
        self.client_ribbon = ctk.CTkFrame(self.ribbon_frame, fg_color="transparent")

        lbl = ctk.CTkLabel(self.client_ribbon, text="Target Server Address:", font=ctk.CTkFont(size=11, weight="bold"), text_color="#78889b")
        lbl.pack(side="left", padx=(4, 4))

        self.entry_client_server_addr = ctk.CTkEntry(
            self.client_ribbon,
            width=200,
            height=28,
            font=ctk.CTkFont(family="monospace", size=11),
            fg_color="#ffffff",
            border_color="#dce3eb",
            text_color="#162d43",
        )
        self.entry_client_server_addr.insert(0, "127.0.0.1:8080")
        self.entry_client_server_addr.pack(side="left", padx=4)

        btn_ping = ctk.CTkButton(
            self.client_ribbon,
            text="⚡ Ping Server",
            width=95,
            height=28,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#0c8c80",
            hover_color="#087b70",
            text_color="#ffffff",
            command=self._ping_client_server,
        )
        btn_ping.pack(side="left", padx=6)

        self.lbl_client_ping_status = ctk.CTkLabel(
            self.client_ribbon,
            text="● Not checked",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#78889b",
        )
        self.lbl_client_ping_status.pack(side="left", padx=6)

    # --- 4x KPI METRIC CARDS ROW ---

    def _build_kpi_row(self):
        self.kpi_frame = ctk.CTkFrame(self.main_container, fg_color="transparent", height=86)
        self.kpi_frame.pack(fill="x", padx=16, pady=(12, 10))
        self.kpi_frame.pack_propagate(False)

        # 4 Metric Cards
        self.kpi_card_1 = self._create_kpi_card("PORT STATUS / TARGET", "Port 8080 Ready", "192.168.1.10 Bound", "#0c8c80")
        self.kpi_card_2 = self._create_kpi_card("FL ROUNDS PROGRESS", "Round 0 / 40", "Local Epochs: 5 · Min Clients: 12", "#2563eb")
        self.kpi_card_3 = self._create_kpi_card("BEST TEST ACCURACY (BTA)", "0.00 %", "Current TA: 0.00%", "#0b8a5a")
        self.kpi_card_4 = self._create_kpi_card("CURRENT LOSS", "0.0000", "Eval Loss: —", "#d79338")

    def _create_kpi_card(self, title: str, default_val: str, default_sub: str, accent_color: str):
        card = ctk.CTkFrame(self.kpi_frame, fg_color="#ffffff", corner_radius=8, border_width=1, border_color="#e8edf2")
        card.pack(side="left", fill="both", expand=True, padx=4)

        # Accent top bar
        bar = ctk.CTkFrame(card, height=3, corner_radius=1, fg_color=accent_color)
        bar.pack(anchor="w", padx=14, pady=(6, 2))
        bar.configure(width=36)

        lbl_title = ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=9, weight="bold"), text_color="#8293a4")
        lbl_title.pack(anchor="w", padx=14, pady=(2, 0))

        lbl_val = ctk.CTkLabel(card, text=default_val, font=ctk.CTkFont(size=18, weight="bold"), text_color="#162d43")
        lbl_val.pack(anchor="w", padx=14, pady=(0, 0))

        lbl_sub = ctk.CTkLabel(card, text=default_sub, font=ctk.CTkFont(size=10), text_color="#78889b")
        lbl_sub.pack(anchor="w", padx=14, pady=(0, 6))

        # Store labels on card object for dynamic updates
        card.lbl_title = lbl_title
        card.lbl_val = lbl_val
        card.lbl_sub = lbl_sub
        return card

    def _bind_kpi_reactive_inputs(self):
        """Bind dynamic text changes in Server and Client forms to update KPI cards immediately."""
        for key in ("rounds", "local_epochs", "min_clients"):
            widget = self.server_ctrl.fields.get(key)
            if widget and hasattr(widget, "bind"):
                widget.bind("<KeyRelease>", self._sync_kpi_display)

        for key in ("local_epochs", "client_id"):
            widget = self.client_ctrl.fields.get(key)
            if widget and hasattr(widget, "bind"):
                widget.bind("<KeyRelease>", self._sync_kpi_display)

    def _sync_kpi_display(self, event=None):
        """Synchronize KPI cards with active view parameters and running state."""
        if self.active_view == "server":
            self.kpi_card_4.lbl_title.configure(text="CURRENT LOSS")
            if not self.server_runner.is_running():
                rounds_val = self.server_ctrl.fields.get("rounds").get().strip() or "40"
                epochs_val = self.server_ctrl.fields.get("local_epochs").get().strip() or "5"
                min_c_val = self.server_ctrl.fields.get("min_clients").get().strip() or "12"
                self.kpi_card_2.lbl_title.configure(text="FL ROUNDS PROGRESS")
                self.kpi_card_2.lbl_val.configure(text=f"Round 0 / {rounds_val}")
                self.kpi_card_2.lbl_sub.configure(text=f"Local Epochs: {epochs_val} · Min Clients: {min_c_val}")
        else:
            self.kpi_card_4.lbl_title.configure(text="CURRENT TRAIN LOSS")
            if not self.client_runner.is_running():
                epochs_val = self.client_ctrl.fields.get("local_epochs").get().strip() or "5"
                cid_val = self.client_ctrl.fields.get("client_id").get().strip() or "0"
                self.kpi_card_2.lbl_title.configure(text="CLIENT TRAINING PROGRESS")
                self.kpi_card_2.lbl_val.configure(text=f"Client Node #{cid_val}")
                self.kpi_card_2.lbl_sub.configure(text=f"Local Epochs: {epochs_val}")

    # --- WORKSPACES ---

    def _build_server_workspace(self):
        self.workspace_server = ctk.CTkFrame(self.workspace_body, fg_color="transparent")

        # Left Column: Server Parameter Configuration Form (width 390)
        self.server_ctrl = ServerTab(
            self.workspace_server,
            width=390,
            on_start=self._start_server,
            on_stop=self._stop_server,
            on_export=self._export_server_results,
        )
        self.server_ctrl.pack(side="left", fill="both", padx=(0, 6), pady=0)

        # Right Column: Visualizer (Top) + Log Console (Bottom)
        right_col = ctk.CTkFrame(self.workspace_server, fg_color="transparent")
        right_col.pack(side="right", fill="both", expand=True, padx=(6, 0), pady=0)

        self.server_plot = DualMetricPlotCanvas(right_col, mode="server")
        self.server_plot.pack(fill="both", expand=True, pady=(0, 6))

        self.server_console = LiveLogConsole(right_col, title="SERVER LIVE LOG CONSOLE")
        self.server_console.pack(fill="both", expand=True, pady=(6, 0))

    def _build_client_workspace(self):
        self.workspace_client = ctk.CTkFrame(self.workspace_body, fg_color="transparent")

        # Left Column: Client Parameters & Partition Inspector (width 390)
        self.client_ctrl = ClientTab(
            self.workspace_client,
            width=390,
            on_start=self._start_client,
            on_stop=self._stop_client,
            on_clear=self._clear_client_plot,
            get_python_exe=lambda: self.active_venv,
        )
        self.client_ctrl.pack(side="left", fill="both", padx=(0, 6), pady=0)

        # Right Column: Client Visualizer (Top) + Client Console (Bottom)
        right_col = ctk.CTkFrame(self.workspace_client, fg_color="transparent")
        right_col.pack(side="right", fill="both", expand=True, padx=(6, 0), pady=0)

        self.client_plot = DualMetricPlotCanvas(right_col, mode="client")
        self.client_plot.pack(fill="both", expand=True, pady=(0, 6))

        self.client_console = LiveLogConsole(right_col, title="CLIENT LIVE LOG CONSOLE")
        self.client_console.pack(fill="both", expand=True, pady=(6, 0))

    def _switch_view(self, mode: str):
        self.active_view = mode
        if mode == "server":
            self.btn_nav_server.configure(fg_color="#1a3c48", text_color="#a7e6d9")
            self.btn_nav_client.configure(fg_color="transparent", text_color="#96aabe")
            self.lbl_breadcrumb.configure(text="FL Platform  ›  FL Server  ›  FL Server Control & Live Monitor")

            self.client_ribbon.pack_forget()
            self.server_ribbon.pack(side="right")

            self.workspace_client.pack_forget()
            self.workspace_server.pack(fill="both", expand=True)

            self.kpi_card_1.lbl_val.configure(text="Port 8080 Ready")
            self.kpi_card_1.lbl_sub.configure(text="Server IP: " + (self.cmb_server_ip.get().split("(")[-1].replace(")", "") if "(" in self.cmb_server_ip.get() else self.cmb_server_ip.get()))
        else:
            self.btn_nav_client.configure(fg_color="#1a3c48", text_color="#a7e6d9")
            self.btn_nav_server.configure(fg_color="transparent", text_color="#96aabe")
            self.lbl_breadcrumb.configure(text="FL Platform  ›  FL Client  ›  FL Client Node & Live Monitor")

            self.server_ribbon.pack_forget()
            self.client_ribbon.pack(side="right")

            self.workspace_server.pack_forget()
            self.workspace_client.pack(fill="both", expand=True)

            addr = self.entry_client_server_addr.get().strip() or "127.0.0.1:8080"
            self.kpi_card_1.lbl_val.configure(text="Client Node")
            self.kpi_card_1.lbl_sub.configure(text=f"Target: {addr}")

        self._sync_kpi_display()

    # --- TOPBAR ACTIONS ---

    def _copy_server_address(self):
        val = self.cmb_server_ip.get()
        ip = val.split("(")[-1].replace(")", "").strip() if "(" in val else val.strip()
        port = self.entry_server_port.get().strip() or "8080"
        addr = f"{ip}:{port}"
        self.clipboard_clear()
        self.clipboard_append(addr)
        self.lbl_server_port_status.configure(text=f"Copied: {addr}", text_color="#0c8c80")

    def _clean_server_port(self):
        try:
            port = int(self.entry_server_port.get().strip())
            killed = kill_processes_on_port(port)
            if killed:
                self.lbl_server_port_status.configure(text=f"Killed PIDs: {killed}", text_color="#0b8a5a")
            else:
                self.lbl_server_port_status.configure(text="Port is free", text_color="#0b8a5a")
        except Exception as e:
            self.lbl_server_port_status.configure(text=f"Error: {e}", text_color="#ba4c4c")

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
            self.lbl_client_ping_status.configure(text=f"● {res['message']}", text_color="#0b8a5a")
        else:
            self.lbl_client_ping_status.configure(text=f"● {res['message']}", text_color="#ba4c4c")

    # --- VENV MANAGEMENT ---

    def _change_venv(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select Python Interpreter for FL",
            filetypes=[("Python Executable", "python.exe"), ("All Files", "*.*")],
        )
        if path and os.path.isfile(path):
            self.active_venv = path
            self.server_console.append_log(f"[CONFIG] Active interpreter updated to: {path}\n")
            self.client_console.append_log(f"[CONFIG] Active interpreter updated to: {path}\n")

    def _test_venv(self):
        VenvDiagnosticModal(self, self.active_venv)

    # --- PROCESS EVENT HANDLERS ---

    def _start_server(self, partial_cmd: List[str]):
        python_exe = self.active_venv
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
        self.after(0, self._sync_kpi_display)

    def _export_server_results(self):
        self.server_plot.export_plot_image()

    def _on_server_log(self, line: str):
        self.server_console.append_log(line)
        self.metric_parser.parse_line(line, source="server")

    def _on_server_exit(self, code: int):
        self.server_console.append_log(f"\n[SERVER TERMINATED] Process exited with code {code}.\n")
        self.after(0, lambda: (self.server_ctrl.set_running_state(False), self._sync_kpi_display()))

        # Check if client_full_histories.json was saved in the latest results directory
        try:
            import glob
            import json
            cand_files = sorted(
                glob.glob(str(PROJECT_ROOT / "results" / "federated" / "*" / "*" / "client_full_histories.json")),
                key=os.path.getmtime,
                reverse=True,
            )
            if cand_files:
                with open(cand_files[0], "r", encoding="utf-8") as f:
                    data = json.load(f)
                    histories = data.get("histories", {})
                    dists = data.get("distributions", {})
                    eval_hists = data.get("eval_histories", {})
                    for cid_str, hist in histories.items():
                        cid = int(cid_str)
                        pkg = {
                            "client_id": cid,
                            "rounds": len(hist),
                            "samples": hist[-1].get("num_samples", 0) if hist else 0,
                            "distribution": dists.get(cid_str, {}),
                            "history": hist,
                            "eval_history": eval_hists.get(cid_str, []),
                        }
                        self.after(0, lambda p=pkg: self.server_plot.add_client_package(p))
        except Exception as e:
            self.server_console.append_log(f"[INSPECTOR] Could not load post-run client histories: {e}\n")

    def _start_client(self, partial_cmd: List[str]):
        python_exe = self.active_venv
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
        self.after(0, self._sync_kpi_display)

    def _clear_client_plot(self):
        self.client_plot.clear()

    def _on_client_log(self, line: str):
        self.client_console.append_log(line)
        self.metric_parser.parse_line(line, source="client")

    def _on_client_exit(self, code: int):
        self.client_console.append_log(f"\n[CLIENT TERMINATED] Process exited with code {code}.\n")
        self.after(0, lambda: (self.client_ctrl.set_running_state(False), self._sync_kpi_display()))

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

            # Dynamic KPI updates
            rounds_cfg = self.server_ctrl.fields.get("rounds")
            total_r = rounds_cfg.get().strip() if rounds_cfg else "40"
            self.after(0, lambda: self.kpi_card_2.lbl_val.configure(text=f"Round {r} / {total_r}"))
            if bta is not None:
                bta_pct = bta * 100 if bta <= 1.0 else bta
                self.after(0, lambda: self.kpi_card_3.lbl_val.configure(text=f"{bta_pct:.2f} %"))
            if ta is not None:
                ta_pct = ta * 100 if ta <= 1.0 else ta
                self.after(0, lambda: self.kpi_card_3.lbl_sub.configure(text=f"Current TA: {ta_pct:.2f}%"))
            if loss is not None:
                self.after(0, lambda: self.kpi_card_4.lbl_val.configure(text=f"{loss:.4f}"))

        elif source == "server_client_bundle":
            self.after(0, lambda: self.server_plot.add_client_package(payload))

        elif source == "client_round_done":
            r = payload.get("round", 1)
            loss = payload.get("train_loss")
            acc = payload.get("train_acc")
            self.after(0, lambda: self.client_plot.update_client_round(r, train_loss=loss, train_acc=acc))

            # Dynamic KPI updates for client
            self.after(0, lambda: self.kpi_card_2.lbl_val.configure(text=f"Round {r}"))
            if acc is not None:
                acc_pct = acc * 100 if acc <= 1.0 else acc
                self.after(0, lambda: self.kpi_card_3.lbl_val.configure(text=f"{acc_pct:.2f} %"))
            if loss is not None:
                self.after(0, lambda: self.kpi_card_4.lbl_val.configure(text=f"{loss:.4f}"))

        elif source == "client_epoch":
            epoch = payload.get("epoch", 1)
            loss = payload.get("train_loss")
            acc = payload.get("train_acc")
            is_reset = payload.get("is_new_round", False) or (epoch == 1)
            self.after(0, lambda: self.client_plot.update_client_epoch(epoch, loss, acc, is_new_round=is_reset))

        elif source == "client_eval_done":
            acc = payload.get("accuracy")
            r = self.client_plot.client_round_data["rounds"][-1] if self.client_plot.client_round_data["rounds"] else 1
            self.after(0, lambda: self.client_plot.update_client_round(r, ta=acc))

        elif source == "client_eval":
            ta = payload.get("ta")
            bta = payload.get("bta")
            r = self.client_plot.client_round_data["rounds"][-1] if self.client_plot.client_round_data["rounds"] else 1
            self.after(0, lambda: self.client_plot.update_client_round(r, ta=ta, bta=bta))

    def _on_telemetry_update(self, info: Dict[str, Any]):
        cpu = info.get("cpu_percent", 0.0)
        ram = info.get("ram_used_gb", 0.0)
        self.lbl_sidebar_telemetry.configure(text=f"CPU: {cpu:.1f}% | RAM: {ram:.1f} GB")
        self.server_console.update_telemetry(info)
        self.client_console.update_telemetry(info)

    def _on_close(self):
        """Cleanup processes and stop telemetry before quitting."""
        self.telemetry.stop()
        self.server_runner.stop()
        self.client_runner.stop()
        self.destroy()


def main():
    app = FLOrchestratorDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
