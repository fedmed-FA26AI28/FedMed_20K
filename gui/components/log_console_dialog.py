import tkinter as tk
from typing import Callable, Optional

import customtkinter as ctk

from gui import theme as T
from gui.components.log_console import LiveLogConsole


class LiveLogConsoleDialog(ctk.CTkToplevel):
    """Floating non-blocking Live Console popup accessible from any tab via the topbar."""

    def __init__(
        self,
        master,
        on_source_change: Optional[Callable[[str], None]] = None,
        on_visibility_change: Optional[Callable[[bool], None]] = None,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self._on_visibility_change = on_visibility_change
        self._is_open = False

        self.title("FedMedAI — Live Orchestration Console")
        self.geometry("920x520")
        self.minsize(640, 360)
        self.configure(fg_color=T.BACKGROUND)

        # Hide immediately at startup so it only appears when user opens Live Console
        self.withdraw()
        self.protocol("WM_DELETE_WINDOW", self.hide)

        self.console = LiveLogConsole(
            self,
            title="LIVE CONSOLE",
            show_source_switcher=True,
            on_source_change=on_source_change,
        )
        self.console.pack(fill="both", expand=True, padx=10, pady=10)

        # Convenience attribute forwarding
        self.seg_source = getattr(self.console, "seg_source", None)
        self.lbl_auto = getattr(self.console, "lbl_auto", None)

    def is_open(self) -> bool:
        return self._is_open

    def show(self):
        """Show and bring the floating console popup to the front."""
        try:
            self.deiconify()
            self.lift()
            self.focus_set()
        except tk.TclError:
            pass
        self._is_open = True
        if self._on_visibility_change:
            self._on_visibility_change(True)

    def hide(self):
        """Hide the floating console popup while keeping all buffered logs intact."""
        try:
            self.withdraw()
        except tk.TclError:
            pass
        self._is_open = False
        if self._on_visibility_change:
            self._on_visibility_change(False)

    def toggle(self):
        """Toggle visibility of the floating console popup."""
        if self._is_open:
            self.hide()
        else:
            self.show()

    def append_log(self, text: str):
        self.console.append_log(text)

    def update_telemetry(self, info: dict):
        self.console.update_telemetry(info)

    def clear(self):
        self.console.clear()
