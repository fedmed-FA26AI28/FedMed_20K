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

from gui import theme as T
from gui.components.server_tab import ServerTab
from gui.components.client_tab import ClientTab
from gui.components.plot_canvas import DualMetricPlotCanvas
from gui.components.log_console import LiveLogConsole
from gui.runners.process_runner import SubprocessRunner
from gui.runners.metric_parser import MetricStreamParser
from gui.telemetry import HardwareTelemetryMonitor, ping_server
from gui.config_manager import get_local_ip_addresses, check_port_in_use, kill_processes_on_port, get_default_venv_path
from gui.components.env_banner import VenvDiagnosticModal
from gui.components.icons import sidebar_icon


class FLOrchestratorDashboard(ctk.CTk):
    """Main Application Window for Unified Federated Learning Orchestrator & Live Monitor."""

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("Dark")

        self.title("FL Orchestrator | Federated Learning Operations")
        self.geometry("1480x940")
        self.minsize(1180, 780)
        self.configure(fg_color=T.BACKGROUND)

        # Active Virtual Environment Interpreter
        self.active_venv = get_default_venv_path()

        # Active Navigation View: "server" or "client"
        self.active_view = "server"
        self._sidebar_visible = True
        self._role_metrics = {"server": {}, "client": {}}

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

    def _toggle_theme(self):
        """Recolor existing widgets without replacing forms, logs or live run state."""
        ctk.set_appearance_mode("Light" if ctk.get_appearance_mode() == "Dark" else "Dark")

    def _toggle_sidebar(self):
        """Give the active workspace more room without rebuilding its contents."""
        self._sidebar_visible = not self._sidebar_visible
        if self._sidebar_visible:
            self.sidebar.pack(side="left", fill="y", before=self.main_container)
            self.btn_sidebar_reopen.pack_forget()
        else:
            self.sidebar.pack_forget()
            self.btn_sidebar_reopen.pack(side="left", padx=(14, 0), before=self.lbl_breadcrumb)

    def _build_layout(self):
        # 1. Fixed navigation sidebar
        self.sidebar = ctk.CTkFrame(self, width=210, corner_radius=0, fg_color=T.SIDEBAR)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Brand Header
        brand_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand_frame.pack(fill="x", padx=14, pady=(14, 12))

        brand_top = ctk.CTkFrame(brand_frame, fg_color="transparent")
        brand_top.pack(fill="x")

        self._sidebar_icon = sidebar_icon()
        self.btn_sidebar_toggle = ctk.CTkButton(
            brand_top, text="", image=self._sidebar_icon,
            width=26, height=26, corner_radius=6, border_spacing=0,
            fg_color=T.SURFACE, hover_color=T.BORDER,
            command=self._toggle_sidebar,
        )
        self.btn_sidebar_toggle.pack(side="left", padx=(0, 8))

        lbl_brand_fl = ctk.CTkLabel(
            brand_top,
            text="FL",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=T.TEXT,
        )
        lbl_brand_fl.pack(side="left")

        lbl_brand_orch = ctk.CTkLabel(
            brand_top,
            text="Orchestrater",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=T.ACCENT,
        )
        lbl_brand_orch.pack(side="left", padx=(3, 0))

        lbl_subtitle = ctk.CTkLabel(
            brand_frame,
            text="FEDERATED LEARNING",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color=T.MUTED,
        )
        lbl_subtitle.pack(anchor="w", padx=(38, 0), pady=(3, 0))

        # Navigation Label
        lbl_nav = ctk.CTkLabel(
            self.sidebar,
            text="WORKSPACES",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color=T.MUTED,
        )
        lbl_nav.pack(anchor="w", padx=16, pady=(18, 6))

        # Navigation Buttons
        self.btn_nav_server = ctk.CTkButton(
            self.sidebar,
            text="  FL Server",
            height=38,
            anchor="w",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=T.SELECTED,
            hover_color=T.SURFACE,
            text_color=T.SELECTED_TEXT,
            command=lambda: self._switch_view("server"),
        )
        self.btn_nav_server.pack(fill="x", padx=12, pady=3)

        self.btn_nav_client = ctk.CTkButton(
            self.sidebar,
            text="  Client Node",
            height=38,
            anchor="w",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="transparent",
            hover_color=T.SURFACE,
            text_color=T.MUTED,
            command=lambda: self._switch_view("client"),
        )
        self.btn_nav_client.pack(fill="x", padx=12, pady=3)

        # Bottom Sidebar Box
        bottom_box = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        bottom_box.pack(side="bottom", fill="x", padx=14, pady=16)

        sep = ctk.CTkFrame(bottom_box, height=1, fg_color=T.BORDER)
        sep.pack(fill="x", pady=(0, 10))

        self.lbl_sidebar_telemetry = ctk.CTkLabel(
            bottom_box,
            text="CPU: 0.0% | RAM: 0.0 GB",
            font=ctk.CTkFont(family="monospace", size=9),
            text_color=T.MUTED,
        )
        self.lbl_sidebar_telemetry.pack(anchor="w", padx=2, pady=(0, 8))

        # Change/Test Venv Buttons
        venv_btns = ctk.CTkFrame(bottom_box, fg_color="transparent")
        venv_btns.pack(fill="x", pady=(2, 0))

        btn_ch_venv = ctk.CTkButton(
            venv_btns,
            text="Change Venv",
            height=26,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=T.BORDER,
            hover_color=T.SURFACE,
            text_color=T.TEXT,
            command=self._change_venv,
        )
        btn_ch_venv.pack(side="left", fill="x", expand=True, padx=(0, 3))

        btn_t_venv = ctk.CTkButton(
            venv_btns,
            text="Test",
            height=26,
            width=50,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=T.PRIMARY,
            hover_color=T.PRIMARY_HOVER,
            text_color=T.ON_ACCENT,
            command=self._test_venv,
        )
        btn_t_venv.pack(side="right", padx=(3, 0))

        # 2. Main workspace
        self.main_container = ctk.CTkFrame(self, fg_color=T.BACKGROUND, corner_radius=0)
        self.main_container.pack(side="right", fill="both", expand=True)

        # 2A. Breadcrumb and network controls (Single compact row matching reference)
        self.topbar = ctk.CTkFrame(self.main_container, height=52, corner_radius=0, fg_color=T.PANEL)
        self.topbar.pack(fill="x")
        self.topbar.pack_propagate(False)

        topbar_border = ctk.CTkFrame(self.main_container, height=1, corner_radius=0, fg_color=T.BORDER)
        topbar_border.pack(fill="x")

        self.btn_sidebar_reopen = ctk.CTkButton(
            self.topbar, text="", image=self._sidebar_icon,
            width=26, height=26, corner_radius=6, border_spacing=0,
            fg_color=T.SURFACE, hover_color=T.BORDER,
            command=self._toggle_sidebar,
        )

        # Topbar Left: Breadcrumb
        self.lbl_breadcrumb = ctk.CTkLabel(
            self.topbar,
            text="FL Platform  ›  FL Server  ›  FL Server Control & Live Monitor",
            font=ctk.CTkFont(size=11),
            text_color=T.MUTED,
        )
        self.lbl_breadcrumb.pack(side="left", padx=16)

        # Topbar Right: Theme button & Quick Stop
        self.theme_button = ctk.CTkButton(
            self.topbar, text="◐", width=28, height=28, corner_radius=14,
            font=ctk.CTkFont(family="Segoe UI Symbol", size=18),
            fg_color="transparent", hover_color=T.SURFACE, text_color=T.TEXT,
            command=self._toggle_theme,
        )
        self.theme_button.pack(side="right", padx=(4, 16))

        self.btn_quick_stop = ctk.CTkButton(
            self.topbar, text="Stop Server", width=90, height=28,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=T.DANGER, hover_color=T.DANGER_HOVER, text_color=T.ON_ACCENT,
            command=self._stop_active_role, state="disabled",
        )
        self.btn_quick_stop.pack(side="right", padx=4)

        # Topbar Right: Dynamic Host / Target Ribbon
        self.ribbon_frame = ctk.CTkFrame(self.topbar, fg_color="transparent")
        self.ribbon_frame.pack(side="right", padx=(0, 8))

        self._build_server_topbar_ribbon(self.ribbon_frame)
        self._build_client_topbar_ribbon(self.ribbon_frame)

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
        self._refresh_workspace_state()

    # --- TOPBAR RIBBONS ---

    def _build_server_topbar_ribbon(self, parent):
        self.server_ribbon = ctk.CTkFrame(parent, fg_color="transparent")

        lbl = ctk.CTkLabel(self.server_ribbon, text="Host IP:", font=ctk.CTkFont(size=11, weight="bold"), text_color=T.MUTED)
        lbl.pack(side="left", padx=(4, 4))

        self.server_ip_list = get_local_ip_addresses()
        ip_options = [f"{label} ({ip})" for label, ip in self.server_ip_list]
        self.cmb_server_ip = ctk.CTkComboBox(
            self.server_ribbon,
            values=ip_options if ip_options else ["127.0.0.1"],
            width=210,
            height=28,
            font=ctk.CTkFont(size=11),
            fg_color=T.PANEL,
            border_color=T.BORDER,
            text_color=T.TEXT,
        )
        self.cmb_server_ip.pack(side="left", padx=4)

        lbl_port = ctk.CTkLabel(self.server_ribbon, text="Port:", font=ctk.CTkFont(size=11, weight="bold"), text_color=T.MUTED)
        lbl_port.pack(side="left", padx=(6, 2))

        self.entry_server_port = ctk.CTkEntry(
            self.server_ribbon,
            width=60,
            height=28,
            font=ctk.CTkFont(family="monospace", size=11),
            fg_color=T.PANEL,
            border_color=T.BORDER,
            text_color=T.TEXT,
        )
        self.entry_server_port.insert(0, "8080")
        self.entry_server_port.pack(side="left", padx=4)

        btn_copy = ctk.CTkButton(
            self.server_ribbon,
            text="📋 Copy",
            width=70,
            height=28,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=T.PANEL,
            hover_color=T.SURFACE,
            text_color=T.TEXT,
            border_width=1,
            border_color=T.BORDER,
            command=self._copy_server_address,
        )
        btn_copy.pack(side="left", padx=6)

        self.lbl_server_port_status = ctk.CTkLabel(
            self.server_ribbon,
            text="● Port 8080 Ready",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=T.SUCCESS,
        )
        self.lbl_server_port_status.pack(side="left", padx=6)

        btn_clean_port = ctk.CTkButton(
            self.server_ribbon,
            text="Clean Port",
            width=75,
            height=26,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=T.PANEL,
            hover_color=T.DANGER_HOVER,
            text_color=T.ERROR,
            border_width=1,
            border_color=T.DANGER,
            command=self._clean_server_port,
        )
        btn_clean_port.pack(side="left", padx=(6, 0))

    def _build_client_topbar_ribbon(self, parent):
        self.client_ribbon = ctk.CTkFrame(parent, fg_color="transparent")

        lbl = ctk.CTkLabel(self.client_ribbon, text="Target Server Address:", font=ctk.CTkFont(size=11, weight="bold"), text_color=T.MUTED)
        lbl.pack(side="left", padx=(4, 4))

        self.entry_client_server_addr = ctk.CTkEntry(
            self.client_ribbon,
            width=200,
            height=28,
            font=ctk.CTkFont(family="monospace", size=11),
            fg_color=T.PANEL,
            border_color=T.BORDER,
            text_color=T.TEXT,
        )
        self.entry_client_server_addr.insert(0, "127.0.0.1:8080")
        self.entry_client_server_addr.pack(side="left", padx=4)

        btn_ping = ctk.CTkButton(
            self.client_ribbon,
            text="⚡ Ping Server",
            width=95,
            height=28,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=T.PRIMARY,
            hover_color=T.PRIMARY_HOVER,
            text_color=T.ON_ACCENT,
            command=self._ping_client_server,
        )
        btn_ping.pack(side="left", padx=6)

        self.lbl_client_ping_status = ctk.CTkLabel(
            self.client_ribbon,
            text="● Not checked",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=T.MUTED,
        )
        self.lbl_client_ping_status.pack(side="left", padx=6)

    # --- 4x KPI METRIC CARDS ROW ---

    def _build_kpi_row(self):
        self.kpi_frame = ctk.CTkFrame(self.main_container, fg_color="transparent", height=116)
        self.kpi_frame.pack(fill="x", padx=20, pady=(14, 16))
        self.kpi_frame.grid_propagate(False)

        self.kpi_frame.grid_rowconfigure(0, weight=1)
        self.kpi_frame.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="kpi")
        self._kpi_column = 0

        # 4 Metric Cards
        self.kpi_card_1 = self._create_kpi_card("NETWORK / TARGET", "Not checked", "Select a host address", T.PRIMARY)
        self.kpi_card_2 = self._create_kpi_card("FL ROUNDS PROGRESS", "Round 0 / 40", "Local Epochs: 5 · Min Clients: 12", T.BLUE)
        self.kpi_card_3 = self._create_kpi_card("BEST TEST ACCURACY (BTA)", "—", "Waiting for evaluation", T.SUCCESS)
        self.kpi_card_4 = self._create_kpi_card("CURRENT LOSS", "—", "Waiting for metrics", T.WARNING)

    def _create_kpi_card(self, title: str, default_val: str, default_sub: str, accent_color: str):
        card = ctk.CTkFrame(self.kpi_frame, fg_color=T.PANEL, corner_radius=12, border_width=1, border_color=T.BORDER)
        card.grid(row=0, column=self._kpi_column, sticky="nsew", padx=4)
        self._kpi_column += 1

        # Accent top bar
        bar = ctk.CTkFrame(card, height=3, corner_radius=1, fg_color=accent_color)
        bar.pack(anchor="w", padx=16, pady=(12, 2))
        bar.configure(width=36)

        lbl_title = ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=10, weight="bold"), text_color=T.ACCENT)
        lbl_title.pack(anchor="w", padx=14, pady=(2, 0))

        lbl_val = ctk.CTkLabel(card, text=default_val, font=ctk.CTkFont(size=22, weight="bold"), text_color=T.TEXT)
        lbl_val.pack(anchor="w", padx=14, pady=(0, 0))

        lbl_sub = ctk.CTkLabel(card, text=default_sub, font=ctk.CTkFont(size=10), text_color=T.MUTED)
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
        """Keep each role's summary independent when navigation changes."""
        mode = self.active_view
        ctrl = self.server_ctrl if mode == "server" else self.client_ctrl
        metrics = self._role_metrics[mode]
        running = getattr(self, mode + "_runner").is_running()
        self.btn_quick_stop.configure(text="Stop Server" if mode == "server" else "Stop Client",
                                      state="normal" if running else "disabled")
        value = lambda key, default: ctrl.fields[key].get().strip() or default
        round_number = metrics.get("round", 0)
        self.kpi_card_2.lbl_title.configure(text="FL ROUNDS PROGRESS" if mode == "server" else "CLIENT TRAINING PROGRESS")
        progress = f"Round {round_number} / {value('rounds', '40')}" if mode == "server" else f"Round {round_number}"
        self.kpi_card_2.lbl_val.configure(text=progress)
        self.kpi_card_2.lbl_sub.configure(text=f"Local epochs: {value('local_epochs', '5')}")
        self.kpi_card_3.lbl_title.configure(text="BEST TEST ACCURACY" if mode == "server" else "CLIENT TRAIN ACCURACY")
        accuracy = metrics.get("bta" if mode == "server" else "train_acc")
        self.kpi_card_3.lbl_val.configure(text="—" if accuracy is None else f"{accuracy * 100 if accuracy <= 1 else accuracy:.2f} %")
        ta = metrics.get("ta")
        self.kpi_card_3.lbl_sub.configure(text="Waiting for evaluation" if ta is None else f"Test accuracy: {ta * 100 if ta <= 1 else ta:.2f}%")
        loss = metrics.get("loss")
        self.kpi_card_4.lbl_title.configure(text="GLOBAL LOSS" if mode == "server" else "CLIENT TRAIN LOSS")
        self.kpi_card_4.lbl_val.configure(text="—" if loss is None else f"{loss:.4f}")
        self.kpi_card_4.lbl_sub.configure(text="Waiting for metrics" if loss is None else f"Latest report · Round {round_number}")
        self.kpi_card_1.lbl_title.configure(text="NETWORK / TARGET")
        self.kpi_card_1.lbl_val.configure(text=f"Port {self.entry_server_port.get().strip() or '8080'}" if mode == "server" else f"Client #{value('client_id', '0')}")
        address = self.cmb_server_ip.get() if mode == "server" else self.entry_client_server_addr.get()
        self.kpi_card_1.lbl_sub.configure(text=address, wraplength=195)

    # --- WORKSPACES ---

    def _build_server_workspace(self):
        self.workspace_server = ctk.CTkFrame(self.workspace_body, fg_color="transparent")
        self.server_monitor = self._build_monitor(self.workspace_server)
        form_parent, analytics = self._build_configuration(self.server_monitor)
        self.server_ctrl = ServerTab(form_parent, on_start=self._start_server,
                                     on_stop=self._stop_server, on_export=self._export_server_results)
        self.server_ctrl.pack(fill="both", expand=True)
        self.server_ctrl.set_running_state(False)
        self.server_plot = DualMetricPlotCanvas(analytics, mode="server")
        self.server_plot.pack(fill="both", expand=True)
        self.server_console = LiveLogConsole(self.server_monitor.tab("Live Terminal"), title="SERVER LIVE TERMINAL")
        self.server_console.pack(fill="both", expand=True)

    def _build_client_workspace(self):
        self.workspace_client = ctk.CTkFrame(self.workspace_body, fg_color="transparent")
        self.client_monitor = self._build_monitor(self.workspace_client)
        form_parent, analytics = self._build_configuration(self.client_monitor)
        self.client_ctrl = ClientTab(form_parent, on_start=self._start_client,
                                     on_stop=self._stop_client, on_clear=self._clear_client_plot,
                                     get_python_exe=lambda: self.active_venv)
        self.client_ctrl.pack(fill="both", expand=True)
        self.client_ctrl.set_running_state(False)
        self.client_plot = DualMetricPlotCanvas(analytics, mode="client")
        self.client_plot.pack(fill="both", expand=True)
        self.client_console = LiveLogConsole(self.client_monitor.tab("Live Terminal"), title="CLIENT LIVE TERMINAL")
        self.client_console.pack(fill="both", expand=True)

    def _build_configuration(self, monitor):
        page = monitor.tab("Configuration & Analytics")
        columns = ctk.CTkFrame(page, fg_color="transparent")
        columns.pack(fill="both", expand=True)
        columns.grid_columnconfigure(0, weight=0, minsize=350)
        columns.grid_columnconfigure(1, weight=1)
        columns.grid_rowconfigure(0, weight=1)
        form = ctk.CTkFrame(columns, width=350, fg_color="transparent")
        form.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        form.pack_propagate(False)
        analytics = ctk.CTkFrame(columns, fg_color="transparent")
        analytics.grid(row=0, column=1, sticky="nsew")
        return form, analytics

    def _refresh_workspace_state(self):
        self._sync_kpi_display()
        self._on_workspace_tab_changed()
        self._workspace_refresh_id = self.after(1000, self._refresh_workspace_state)

    def _build_monitor(self, parent):
        monitor = ctk.CTkTabview(
            parent, fg_color="transparent", corner_radius=12, anchor="w",
            segmented_button_fg_color=T.SURFACE,
            segmented_button_selected_color=T.PRIMARY,
            segmented_button_selected_hover_color=T.PRIMARY_HOVER,
            segmented_button_unselected_color=T.SURFACE,
            segmented_button_unselected_hover_color=T.BORDER,
            text_color=T.TEXT, command=self._on_workspace_tab_changed,
        )
        monitor.pack(fill="both", expand=True)
        names = ("Configuration & Analytics", "Live Terminal")
        for name in names:
            monitor.add(name)
        return monitor

    def _on_workspace_tab_changed(self):
        tab = getattr(self, self.active_view + "_monitor").get()
        role = "FL Server" if self.active_view == "server" else "Client Node"
        role_label = "FL Server Control & Live Monitor" if self.active_view == "server" else "Client Node Execution & Analytics"
        self.lbl_breadcrumb.configure(text=f"FL Platform  ›  {role}  ›  {role_label}")

    def _switch_view(self, mode: str):
        self.active_view = mode
        for role in ("server", "client"):
            selected = role == mode
            getattr(self, "btn_nav_" + role).configure(
                fg_color=T.SELECTED if selected else "transparent",
                text_color=T.SELECTED_TEXT if selected else T.MUTED)
            ribbon = getattr(self, role + "_ribbon")
            if selected:
                ribbon.pack(side="left")
            else:
                ribbon.pack_forget()
            workspace = getattr(self, "workspace_" + role)
            if selected:
                workspace.pack(fill="both", expand=True)
            else:
                workspace.pack_forget()
        self._sync_kpi_display()
        self._on_workspace_tab_changed()

    def _stop_active_role(self):
        if self.active_view == "server":
            self._stop_server()
        else:
            self._stop_client()

    # --- TOPBAR ACTIONS ---

    def _copy_server_address(self):
        val = self.cmb_server_ip.get()
        ip = val.split("(")[-1].replace(")", "").strip() if "(" in val else val.strip()
        port = self.entry_server_port.get().strip() or "8080"
        addr = f"{ip}:{port}"
        self.clipboard_clear()
        self.clipboard_append(addr)
        self.lbl_server_port_status.configure(text=f"Copied: {addr}", text_color=T.ACCENT)

    def _clean_server_port(self):
        try:
            port = int(self.entry_server_port.get().strip())
            killed = kill_processes_on_port(port)
            if killed:
                self.lbl_server_port_status.configure(text=f"Killed PIDs: {killed}", text_color=T.SUCCESS)
            else:
                self.lbl_server_port_status.configure(text="Port is free", text_color=T.SUCCESS)
        except Exception as e:
            self.lbl_server_port_status.configure(text=f"Error: {e}", text_color=T.ERROR)

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
            self.lbl_client_ping_status.configure(text=f"● {res['message']}", text_color=T.SUCCESS)
        else:
            self.lbl_client_ping_status.configure(text=f"● {res['message']}", text_color=T.ERROR)

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
        self._role_metrics["server"].clear()

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
        self._role_metrics["client"].clear()

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
        self._role_metrics["client"].clear()

    def _on_client_log(self, line: str):
        self.client_console.append_log(line)
        self.metric_parser.parse_line(line, source="client")

    def _on_client_exit(self, code: int):
        self.client_console.append_log(f"\n[CLIENT TERMINATED] Process exited with code {code}.\n")
        self.after(0, lambda: (self.client_ctrl.set_running_state(False), self._sync_kpi_display()))

    def _on_metric_event(self, source: str, payload: Dict[str, Any]):
        """Dispatch worker metrics to Tk before updating the active role's summary."""
        self.after(0, self._apply_metric_event, source, payload)

    def _apply_metric_event(self, source, payload):
        if source == "server_client_bundle":
            self.server_plot.add_client_package(payload)
            return
        mode = "server" if source == "server" else "client"
        metrics = self._role_metrics[mode]
        r = payload.get("round", metrics.get("round", 1))
        if source == "server":
            keys = ("loss", "bta", "ta")
            self.server_plot.update_server_metrics(r, payload.get("loss"), payload.get("bta"),
                                                   payload.get("bmcta"), payload.get("ta"), payload.get("mcta"))
            metrics.update({key: payload[key] for key in keys if payload.get(key) is not None})
            metrics["round"] = r
        elif source == "client_round_done":
            self.client_plot.update_client_round(r, train_loss=payload.get("train_loss"), train_acc=payload.get("train_acc"))
            for key, source_key in (("loss", "train_loss"), ("train_acc", "train_acc")):
                if payload.get(source_key) is not None:
                    metrics[key] = payload[source_key]
            metrics["round"] = r
        elif source == "client_epoch":
            epoch = payload.get("epoch", 1)
            self.client_plot.update_client_epoch(epoch, payload.get("train_loss"), payload.get("train_acc"),
                                                 is_new_round=payload.get("is_new_round", False) or epoch == 1)
        elif source in ("client_eval_done", "client_eval"):
            rounds = self.client_plot.client_round_data["rounds"]
            r = rounds[-1] if rounds else 1
            ta = payload.get("accuracy") if source == "client_eval_done" else payload.get("ta")
            self.client_plot.update_client_round(r, ta=ta, bta=payload.get("bta"))
            if ta is not None:
                metrics["ta"] = ta
        self._sync_kpi_display()

    def _on_telemetry_update(self, info: Dict[str, Any]):
        cpu = info.get("cpu_percent", 0.0)
        ram = info.get("ram_used_gb", 0.0)
        self.lbl_sidebar_telemetry.configure(text=f"CPU: {cpu:.1f}% | RAM: {ram:.1f} GB")
        self.server_console.update_telemetry(info)
        self.client_console.update_telemetry(info)

    def _on_close(self):
        """Cleanup processes and stop telemetry before quitting."""
        if hasattr(self, "_workspace_refresh_id"):
            self.after_cancel(self._workspace_refresh_id)
        self.telemetry.stop()
        self.server_runner.stop()
        self.client_runner.stop()
        self.destroy()


def main():
    app = FLOrchestratorDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
