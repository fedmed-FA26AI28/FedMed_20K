import tkinter as tk
import customtkinter as ctk
from typing import Optional, Dict, Any
from gui import theme as T
from gui.components.plot_canvas import DualMetricPlotCanvas
from gui.components.log_console import LiveLogConsole
from gui.components.recorded_metrics_table import RecordedMetricsTable


class AnalyticsTab(ctk.CTkFrame):
    """Dedicated Plots and Live Terminal Tab with stacked curve visualizer, recorded metrics table, and dynamic console."""

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.active_log_source = "server"  # "server" or "client"
        self.server_logs = []
        self.client_logs = []

        self._build_layout()

    def _build_layout(self):
        # Two-column layout: Left (Stacked Plot Canvas) + Right (Recorded Metrics Table on top, Live Console on bottom)
        self.columns = ctk.CTkFrame(self, fg_color="transparent")
        self.columns.pack(fill="both", expand=True)

        self.columns.grid_columnconfigure(0, weight=1, uniform="cols")
        self.columns.grid_columnconfigure(1, weight=1, uniform="cols")
        self.columns.grid_rowconfigure(0, weight=1)

        # Left Column: DualMetricPlotCanvas (Vertically stacked curves: Accuracy top, Loss bottom)
        self.left_frame = ctk.CTkFrame(self.columns, fg_color="transparent")
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=0)

        self.plot_canvas = DualMetricPlotCanvas(self.left_frame, mode="server")
        self.plot_canvas.pack(fill="both", expand=True)

        # Right Column: Split into Top (Recorded Metrics Table) + Bottom (Context-Aware Live Console moved down)
        self.right_frame = ctk.CTkFrame(self.columns, fg_color="transparent")
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=0)
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(0, weight=1, uniform="right_col")  # Top: Recorded Metrics Table (Server & Clients)
        self.right_frame.grid_rowconfigure(1, weight=1, uniform="right_col")  # Bottom: Live Log Console (Moved down!)

        # 1. Top Half: Table of Recorded Metrics of Server and Clients
        self.metrics_table = RecordedMetricsTable(self.right_frame)
        self.metrics_table.grid(row=0, column=0, sticky="nsew", padx=0, pady=(0, 4))

        # 2. Bottom Half: Live Log Console with integrated stream switcher (saves vertical space)
        self.console = LiveLogConsole(
            self.right_frame,
            title="LIVE CONSOLE",
            show_source_switcher=True,
            on_source_change=self._on_source_switch,
        )
        self.console.grid(row=1, column=0, sticky="nsew", padx=0, pady=(4, 0))

        # Backward compatibility aliases
        self.console_container = self.console
        self.seg_source = getattr(self.console, "seg_source", None)
        self.lbl_auto = getattr(self.console, "lbl_auto", None)

    def _on_source_switch(self, choice: str):
        self.active_log_source = choice.lower()
        self._reload_console_buffer()

    def set_active_source(self, source: str, auto: bool = True):
        """Switch active console stream (e.g. called automatically when server or client runs)."""
        src = source.lower()
        if src in ("server", "client"):
            self.active_log_source = src
            if hasattr(self.console, "seg_source") and self.console.seg_source:
                self.console.seg_source.set("Server" if src == "server" else "Client")
            if auto and hasattr(self.console, "lbl_auto") and self.console.lbl_auto:
                self.console.lbl_auto.configure(text=f"[Auto: {src.capitalize()}]", text_color=T.SUCCESS)
            self._reload_console_buffer()
        src = source.lower()
        if src in ("server", "client"):
            self.active_log_source = src
            self.seg_source.set("Server" if src == "server" else "Client")
            if auto:
                self.lbl_auto.configure(text=f"[Auto: {src.capitalize()}]", text_color=T.SUCCESS)
            self._reload_console_buffer()

    def append_server_log(self, text: str):
        self.server_logs.append(text)
        if len(self.server_logs) > 3000:
            self.server_logs.pop(0)
        if self.active_log_source == "server":
            self.console.append_log(text)

    def append_client_log(self, text: str):
        self.client_logs.append(text)
        if len(self.client_logs) > 3000:
            self.client_logs.pop(0)
        if self.active_log_source == "client":
            self.console.append_log(text)

    def _reload_console_buffer(self):
        self.console.clear()
        logs = self.server_logs if self.active_log_source == "server" else self.client_logs
        for line in logs[-1000:]:
            self.console.append_log(line)

    def update_telemetry(self, info: Dict[str, Any]):
        self.console.update_telemetry(info)

    def clear(self):
        self.server_logs.clear()
        self.client_logs.clear()
        self.console.clear()
        self.plot_canvas.clear()
        if hasattr(self, "metrics_table"):
            self.metrics_table.clear()
