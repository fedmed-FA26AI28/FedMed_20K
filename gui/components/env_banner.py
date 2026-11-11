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
        self.configure(fg_color="#0f172a")

        self.transient(master)
        self.grab_set()

        # Header
        lbl_hdr = ctk.CTkLabel(
            self,
            text="🔍 Virtual Environment Diagnostics",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#60a5fa",
        )
        lbl_hdr.pack(pady=(16, 4))

        lbl_path = ctk.CTkLabel(
            self,
            text=f"Probing: {python_exe}",
            font=ctk.CTkFont(family="monospace", size=10),
            text_color="#94a3b8",
        )
        lbl_path.pack(pady=(0, 14))

        # Checklist Container
        self.box = ctk.CTkFrame(self, fg_color="#1e293b", corner_radius=8)
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
            fg_color="#334155",
            hover_color="#475569",
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
                "FedMedAI Algorithms & Models",
                "Loaded cleanly" if diag.get("project_modules") else "Import Failed",
                diag.get("project_modules", False),
            ),
        ]

        for title, detail, is_ok in items:
            row = ctk.CTkFrame(self.box, fg_color="transparent")
            row.pack(fill="x", padx=16, pady=6)

            icon = "✓" if is_ok else "✗"
            icon_color = "#34d399" if is_ok else "#f87171"

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
                text_color="#f1f5f9",
            )
            lbl_title.pack(side="left", padx=6)

            lbl_detail = ctk.CTkLabel(
                row,
                text=detail,
                font=ctk.CTkFont(size=11),
                text_color="#94a3b8",
            )
            lbl_detail.pack(side="right")

        if diag.get("error"):
            err_box = ctk.CTkLabel(
                self.box,
                text=f"Notice: {diag['error']}",
                font=ctk.CTkFont(size=10),
                text_color="#f87171",
                wraplength=480,
            )
            err_box.pack(pady=8)


class EnvBanner(ctk.CTkFrame):
    """Top environment banner showing current python interpreter, status badge, and control buttons."""

    def __init__(self, master, on_venv_changed: Optional[Callable[[str], None]] = None, **kwargs):
        super().__init__(master, fg_color="#1e293b", corner_radius=8, height=60, **kwargs)
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
            text="FedMedAI Federated Orchestrator & Live Monitor",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#60a5fa",
        )
        title.pack(anchor="w")

        short_path = self.active_venv
        if len(short_path) > 65:
            short_path = "..." + short_path[-60:]
        self.lbl_path = ctk.CTkLabel(
            left_box,
            text=f"Active Venv: {short_path}",
            font=ctk.CTkFont(family="monospace", size=10),
            text_color="#94a3b8",
        )
        self.lbl_path.pack(anchor="w")

        # Right: Action Buttons
        btn_box = ctk.CTkFrame(self, fg_color="transparent")
        btn_box.pack(side="right", padx=16, pady=6)

        self.btn_change = ctk.CTkButton(
            btn_box,
            text="📁 Change Venv...",
            width=130,
            height=32,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#334155",
            hover_color="#475569",
            command=self._change_venv,
        )
        self.btn_change.pack(side="right", padx=6)

        self.btn_test = ctk.CTkButton(
            btn_box,
            text="🔍 Test Venv",
            width=110,
            height=32,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#2563eb",
            hover_color="#1d4ed8",
            command=self._test_venv,
        )
        self.btn_test.pack(side="right", padx=6)

        # Center: Status Badge
        self.lbl_status = ctk.CTkLabel(
            self,
            text="● Checking Venv...",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#34d399",
            fg_color="#064e3b",
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
            self.lbl_status.configure(text=text, fg_color="#064e3b", text_color="#34d399")
        else:
            self.lbl_status.configure(text="● VENV WARNING: Missing PyTorch/Flower", fg_color="#7f1d1d", text_color="#fca5a5")

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
