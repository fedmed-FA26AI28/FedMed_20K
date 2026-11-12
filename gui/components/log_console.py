import tkinter as tk
import customtkinter as ctk
from typing import Dict, Any, Optional


class LiveLogConsole(ctk.CTkFrame):
    """Real-time scrolling terminal output console with syntax highlighting and hardware telemetry."""

    def __init__(self, master, title: str = "REAL-TIME LOG CONSOLE", **kwargs):
        super().__init__(master, fg_color="#102135", corner_radius=8, border_width=1, border_color="#1e2d42", **kwargs)
        self.autoscroll = tk.BooleanVar(value=True)

        self._build_header(title)
        self._build_text_area()

    def _build_header(self, title: str):
        self.header_frame = ctk.CTkFrame(self, fg_color="#16273b", height=32, corner_radius=6)
        self.header_frame.pack(fill="x", padx=8, pady=(8, 4))

        lbl = ctk.CTkLabel(
            self.header_frame,
            text=f"📟 {title}",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#e2e8f0",
        )
        lbl.pack(side="left", padx=10)

        # Telemetry badge
        self.lbl_telemetry = ctk.CTkLabel(
            self.header_frame,
            text="GPU: Scanning... | VRAM: 0.0 GB",
            font=ctk.CTkFont(family="monospace", size=10, weight="bold"),
            text_color="#2caa9f",
        )
        self.lbl_telemetry.pack(side="left", padx=15)

        # Clear button
        btn_clear = ctk.CTkButton(
            self.header_frame,
            text="Clear Logs",
            width=70,
            height=22,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#1c334d",
            hover_color="#223f5e",
            text_color="#c7d4e0",
            command=self.clear,
        )
        btn_clear.pack(side="right", padx=6)

        # Autoscroll switch
        cb_auto = ctk.CTkCheckBox(
            self.header_frame,
            text="Autoscroll",
            variable=self.autoscroll,
            font=ctk.CTkFont(size=10),
            text_color="#94a3b8",
            fg_color="#1c334d",
            checkmark_color="#0c8c80",
            checkbox_width=18,
            checkbox_height=18,
        )
        cb_auto.pack(side="right", padx=10)

    def _build_text_area(self):
        # We use standard tk.Text with dark configuration for full control over tag coloring
        self.textbox = tk.Text(
            self,
            bg="#0d1726",
            fg="#cbd5e1",
            insertbackground="#60a5fa",
            relief="flat",
            font=("Consolas", 10),
            wrap="word",
        )
        self.scrollbar = ctk.CTkScrollbar(self, command=self.textbox.yview)
        self.textbox.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y", padx=(0, 6), pady=(0, 6))
        self.textbox.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=(0, 6))

        # Setup highlight tags
        self.textbox.tag_config("info", foreground="#60a5fa")
        self.textbox.tag_config("metrics", foreground="#34d399")
        self.textbox.tag_config("round", foreground="#fbbf24")
        self.textbox.tag_config("error", foreground="#f87171")
        self.textbox.tag_config("clients", foreground="#a78bfa")
        self.textbox.tag_config("timestamp", foreground="#64748b")

    def append_log(self, text: str):
        """Thread-safe append of a log line with auto-formatting."""
        self.after(0, self._insert_line, text)

    def _insert_line(self, line: str):
        tag = None
        upper = line.upper()
        if "[ERROR]" in upper or "FAILED" in upper or "TRACEBACK" in upper:
            tag = "error"
        elif "[METRICS" in upper or "BTA:" in upper or "BMCTA:" in upper:
            tag = "metrics"
        elif "[ROUND" in upper or "ROUND " in upper:
            tag = "round"
        elif "[CLIENT" in upper or "CLIENTS" in upper:
            tag = "clients"
        elif "[INFO]" in upper or "[SERVER]" in upper or "[CONNECT]" in upper:
            tag = "info"

        if tag:
            self.textbox.insert(tk.END, line, tag)
        else:
            self.textbox.insert(tk.END, line)

        if self.autoscroll.get():
            self.textbox.see(tk.END)

    def update_telemetry(self, info: Dict[str, Any]):
        """Update hardware telemetry badge."""
        if not info.get("has_gpu"):
            text = "CPU Only Mode"
        else:
            name = info.get("gpu_name") or "GPU"
            if len(name) > 16:
                name = name[:16] + "..."
            used = info.get("vram_used_gb", 0.0)
            total = info.get("vram_total_gb", 0.0)
            text = f"{name} | VRAM: {used}/{total} GB"

        self.after(0, lambda: self.lbl_telemetry.configure(text=text))

    def clear(self):
        self.textbox.delete("1.0", tk.END)
