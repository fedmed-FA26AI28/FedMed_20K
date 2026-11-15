import tkinter as tk
import customtkinter as ctk
from typing import Optional, Dict, Any
from gui import theme as T
from gui.components.plot_canvas import DualMetricPlotCanvas
from gui.components.log_console import LiveLogConsole
from gui.components.recorded_metrics_table import RecordedMetricsTable


class AnalyticsTab(ctk.CTkFrame):
    """Full-width Analytics & Real-Time Visualizer tab (Top: Side-by-Side Plots, Bottom: Recorded Metrics Table)."""

    def __init__(self, master, console_dialog: Optional[Any] = None, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.console_dialog = console_dialog
        self.active_log_source = "server"  # "server" or "client"
        self.server_logs = []
        self.client_logs = []

        self._build_layout()

    def _build_layout(self):
        # Vertical stack: Top 55% side-by-side DualMetricPlotCanvas, Bottom 45% full-width RecordedMetricsTable
        self.main_stack = ctk.CTkFrame(self, fg_color="transparent")
        self.main_stack.pack(fill="both", expand=True)

        self.main_stack.grid_columnconfigure(0, weight=1)
        self.main_stack.grid_rowconfigure(0, weight=55, uniform="analytics_rows")
        self.main_stack.grid_rowconfigure(1, weight=45, uniform="analytics_rows")

        # 1. Top Half: Full-Width Side-by-Side Plot Visualizer
        self.plot_canvas = DualMetricPlotCanvas(
            self.main_stack,
            mode="server",
            on_role_change=self._on_plot_role_switched,
        )
        self.plot_canvas.grid(row=0, column=0, sticky="nsew", padx=0, pady=(0, 6))

        # 2. Bottom Half: Full-Width Recorded Metrics Table
        self.metrics_table = RecordedMetricsTable(self.main_stack)
        self.metrics_table.grid(row=1, column=0, sticky="nsew", padx=0, pady=(6, 0))

        # Wire external LiveLogConsoleDialog (or fallback detached LiveLogConsole for standalone use)
        if self.console_dialog is not None and hasattr(self.console_dialog, "console"):
            self.console = self.console_dialog.console
        else:
            self.console = LiveLogConsole(
                self,
                title="LIVE CONSOLE",
                show_source_switcher=True,
                on_source_change=self._on_source_switch,
            )

        # Backward compatibility aliases
        self.console_container = self.console
        self.seg_source = getattr(self.console, "seg_source", None)
        self.lbl_auto = getattr(self.console, "lbl_auto", None)

    def _on_plot_role_switched(self, role: str):
        self.set_role(role, auto=False, sync_plot=False)

    def _on_source_switch(self, choice: str):
        src = "client" if "client" in choice.lower() else "server"
        self.set_active_source(src, auto=False)

    def set_role(self, role: str, auto: bool = True, sync_plot: bool = True):
        """Synchronize role ('server' or 'client') across plot canvas, metrics table, and live console."""
        target = "client" if role.lower() == "client" else "server"
        if sync_plot and hasattr(self, "plot_canvas"):
            self.plot_canvas.set_mode(target, notify=False)
        if hasattr(self, "metrics_table"):
            self.metrics_table.set_view_mode(target)
        self.set_active_source(target, auto=auto)

    def set_active_source(self, source: str, auto: bool = True):
        """Switch active console stream (e.g. called automatically when server or client runs)."""
        src = source.lower()
        if src not in ("server", "client"):
            return
        self.active_log_source = src
        if hasattr(self, "console") and self.console is not None:
            if hasattr(self.console, "seg_source") and self.console.seg_source:
                self.console.seg_source.set("Server" if src == "server" else "Client")
            if hasattr(self.console, "lbl_auto") and self.console.lbl_auto:
                if auto:
                    self.console.lbl_auto.configure(text=f"[Auto: {src.capitalize()}]", text_color=T.SUCCESS)
                else:
                    self.console.lbl_auto.configure(text=f"[Manual: {src.capitalize()}]", text_color=T.ACCENT)
        self._reload_console_buffer()

    def append_server_log(self, text: str):
        self.server_logs.append(text)
        if len(self.server_logs) > 3000:
            self.server_logs.pop(0)
        if self.active_log_source == "server" and hasattr(self, "console") and self.console is not None:
            self.console.append_log(text)

    def append_client_log(self, text: str):
        self.client_logs.append(text)
        if len(self.client_logs) > 3000:
            self.client_logs.pop(0)
        if self.active_log_source == "client" and hasattr(self, "console") and self.console is not None:
            self.console.append_log(text)

    def _reload_console_buffer(self):
        if not hasattr(self, "console") or self.console is None:
            return
        self.console.clear()
        logs = self.server_logs if self.active_log_source == "server" else self.client_logs
        for line in logs[-1000:]:
            self.console.append_log(line)

    def update_telemetry(self, info: Dict[str, Any]):
        if hasattr(self, "console") and self.console is not None:
            self.console.update_telemetry(info)

    def clear(self):
        self.server_logs.clear()
        self.client_logs.clear()
        if hasattr(self, "console") and self.console is not None:
            self.console.clear()
        self.plot_canvas.clear()
        if hasattr(self, "metrics_table"):
            self.metrics_table.clear()
