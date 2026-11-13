from gui import theme as T
import os
import sys
import customtkinter as ctk
from tkinter import filedialog
from typing import Callable, Optional
from gui.config_manager import test_venv_environment, get_default_venv_path


class VenvDiagnosticModal(ctk.CTkToplevel):
    """Modal dialog displaying comprehensive environment checklist with pass/fail indicators."""

    def __init__(self, master, python_exe: str):
        super().__init__(master)
        self.title("Environment Health Diagnostic")
        self.geometry("560x420")
        self.resizable(False, False)
        self.configure(fg_color=T.SIDEBAR)

        self.transient(master)
        self.grab_set()

        # Header
        lbl_hdr = ctk.CTkLabel(
            self,
            text="🔍 Virtual Environment Diagnostics",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=T.BLUE,
        )
        lbl_hdr.pack(pady=(16, 4))

        lbl_path = ctk.CTkLabel(
            self,
            text=f"Probing: {python_exe}",
            font=ctk.CTkFont(family="monospace", size=10),
            text_color=T.MUTED,
        )
        lbl_path.pack(pady=(0, 14))

        # Checklist Container
        self.box = ctk.CTkFrame(self, fg_color=T.PANEL, corner_radius=12)
        self.box.pack(fill="both", expand=True, padx=20, pady=(0, 16))

        # Run Probe
        diag = test_venv_environment(python_exe)
        self._populate_checklist(diag)

        # Close button
        btn_close = ctk.CTkButton(
            self,
            text="Close",
            width=120,
            command=self.destroy,
            fg_color=T.BORDER,
            hover_color=T.MUTED,
        )
        btn_close.pack(pady=(0, 16))

    def _populate_checklist(self, diag: dict):
        items = [
            ("Python Core", f"v{diag.get('python_version')}", bool(diag.get("python_version"))),
            (
                "PyTorch & Tensor Engine",
                f"v{diag.get('torch_version', 'N/A')}",
                diag.get("torch", False),
            ),
            (
                "CUDA Hardware Acceleration",
                diag.get("cuda_device") or ("Disabled / CPU Only" if diag.get("torch") else "Unavailable"),
                diag.get("cuda_available", False),
            ),
            (
                "Flower FL Framework (flwr)",
                f"v{diag.get('flower_version', 'N/A')}",
                diag.get("flower", False),
            ),
            (
                "MedMNIST Medical Datasets",
                "Available" if diag.get("medmnist") else "Not Installed",
                diag.get("medmnist", False),
            ),
            (
                "FL Algorithms & Models",
                "Loaded cleanly" if diag.get("project_modules") else "Import Failed",
                diag.get("project_modules", False),
            ),
        ]

        for title, detail, is_ok in items:
            row = ctk.CTkFrame(self.box, fg_color="transparent")
            row.pack(fill="x", padx=16, pady=6)

            icon = "✓" if is_ok else "✗"
            icon_color = T.SUCCESS if is_ok else T.ERROR

            lbl_icon = ctk.CTkLabel(
                row,
                text=f"[{icon}]",
                font=ctk.CTkFont(family="monospace", size=14, weight="bold"),
                text_color=icon_color,
                width=30,
            )
            lbl_icon.pack(side="left")

            lbl_title = ctk.CTkLabel(
                row,
                text=title,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=T.SURFACE,
            )
            lbl_title.pack(side="left", padx=6)

            lbl_detail = ctk.CTkLabel(
                row,
                text=detail,
                font=ctk.CTkFont(size=11),
                text_color=T.MUTED,
            )
            lbl_detail.pack(side="right")

        if diag.get("error"):
            err_box = ctk.CTkLabel(
                self.box,
                text=f"Notice: {diag['error']}",
                font=ctk.CTkFont(size=10),
                text_color=T.ERROR,
                wraplength=480,
            )
            err_box.pack(pady=8)


class EnvBanner(ctk.CTkFrame):
    """Top environment banner showing current python interpreter, status badge, and control buttons."""

    def __init__(self, master, on_venv_changed: Optional[Callable[[str], None]] = None, **kwargs):
        super().__init__(master, fg_color=T.PANEL, corner_radius=12, border_width=1, border_color=T.BORDER, height=52, **kwargs)
        self.on_venv_changed = on_venv_changed
        self.active_venv = get_default_venv_path()

        self._build_ui()
        self.refresh_status()

    def _build_ui(self):
        # Left: Title and active path
        left_box = ctk.CTkFrame(self, fg_color="transparent")
        left_box.pack(side="left", padx=16, pady=6)

        title = ctk.CTkLabel(
            left_box,
            text="FL Orchestrator & Live Monitor",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=T.TEXT,
        )
        title.pack(anchor="w")

        short_path = self.active_venv
        if len(short_path) > 65:
            short_path = "..." + short_path[-60:]
        self.lbl_path = ctk.CTkLabel(
            left_box,
            text=f"Active Venv: {short_path}",
            font=ctk.CTkFont(family="monospace", size=10),
            text_color=T.MUTED,
        )
        self.lbl_path.pack(anchor="w")

        # Right: Action Buttons
        btn_box = ctk.CTkFrame(self, fg_color="transparent")
        btn_box.pack(side="right", padx=16, pady=6)

        self.btn_change = ctk.CTkButton(
            btn_box,
            text="📁 Change Venv...",
            width=120,
            height=30,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=T.PANEL,
            hover_color=T.SURFACE,
            text_color=T.TEXT,
            border_width=1,
            border_color=T.BORDER,
            command=self._change_venv,
        )
        self.btn_change.pack(side="right", padx=6)

        self.btn_test = ctk.CTkButton(
            btn_box,
            text="🔍 Test Venv",
            width=100,
            height=30,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=T.PRIMARY,
            hover_color=T.PRIMARY_HOVER,
            text_color=T.ON_ACCENT,
            command=self._test_venv,
        )
        self.btn_test.pack(side="right", padx=6)

        # Center: Status Badge
        self.lbl_status = ctk.CTkLabel(
            self,
            text="● Checking Venv...",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=T.ACCENT,
            fg_color=T.SELECTED,
            corner_radius=12,
            padx=12,
            pady=4,
        )
        self.lbl_status.pack(side="right", padx=12)

    def refresh_status(self):
        diag = test_venv_environment(self.active_venv)
        if diag.get("torch") and diag.get("flower"):
            dev = "CUDA OK" if diag.get("cuda_available") else "CPU"
            text = f"● VENV READY: PyTorch {diag.get('torch_version')} | {dev} | Flower {diag.get('flower_version')}"
            self.lbl_status.configure(text=text, fg_color=T.SUCCESS_SURFACE, text_color=T.SUCCESS)
        else:
            self.lbl_status.configure(text="● VENV WARNING: Missing PyTorch/Flower", fg_color=T.ERROR_SURFACE, text_color=T.ERROR)

    def _test_venv(self):
        VenvDiagnosticModal(self, self.active_venv)

    def _change_venv(self):
        filename = filedialog.askopenfilename(
            title="Select Python Executable for Virtual Environment",
            filetypes=[("Python Executable", "python.exe;python"), ("All Files", "*.*")],
        )
        if filename and os.path.isfile(filename):
            self.active_venv = str(filename)
            short = self.active_venv
            if len(short) > 65:
                short = "..." + short[-60:]
            self.lbl_path.configure(text=f"Active Venv: {short}")
            self.refresh_status()
            if self.on_venv_changed:
                self.on_venv_changed(self.active_venv)
