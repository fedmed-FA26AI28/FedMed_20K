from gui import theme as T
import tkinter as tk
import customtkinter as ctk
from typing import Dict, Any, Optional


class LiveLogConsole(ctk.CTkFrame):
    """Real-time scrolling terminal output console with syntax highlighting and hardware telemetry."""

    def __init__(self, master, title: str = "REAL-TIME LOG CONSOLE", **kwargs):
        super().__init__(master, fg_color=T.SIDEBAR, corner_radius=12, border_width=1, border_color=T.BORDER, **kwargs)
        self.autoscroll = tk.BooleanVar(value=True)
        self.font_size = 10

        self._build_header(title)
        self._build_text_area()

    def _build_header(self, title: str):
        self.header_frame = ctk.CTkFrame(self, fg_color=T.SURFACE, height=32, corner_radius=6)
        self.header_frame.pack(fill="x", padx=8, pady=(8, 4))

        title_row = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        title_row.pack(fill="x", padx=4, pady=(4, 0))
        telemetry_row = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        telemetry_row.pack(fill="x", padx=4)
        controls = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        controls.pack(fill="x", padx=4, pady=(0, 6))
        lbl = ctk.CTkLabel(
            title_row,
            text=f"📟 {title}",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=T.TEXT,
        )
        lbl.pack(side="left", padx=10)

        # Telemetry badge
        self.lbl_telemetry = ctk.CTkLabel(
            telemetry_row,
            text="GPU: Scanning... | VRAM: 0.0 GB",
            font=ctk.CTkFont(family="monospace", size=10, weight="bold"),
            text_color=T.ACCENT,
        )
        self.lbl_telemetry.pack(side="left", padx=15)

        # Clear button
        btn_clear = ctk.CTkButton(
            controls,
            text="Clear Logs",
            width=70,
            height=22,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=T.BORDER,
            hover_color=T.SURFACE,
            text_color=T.TEXT,
            command=self.clear,
        )
        btn_clear.pack(side="right", padx=6)

        # Font zoom controls (A- / 10pt / A+)
        font_box = ctk.CTkFrame(controls, fg_color="transparent")
        font_box.pack(side="right", padx=(4, 6))

        btn_zoom_out = ctk.CTkButton(
            font_box,
            text="A-",
            width=26,
            height=22,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=T.BORDER,
            hover_color=T.SURFACE,
            text_color=T.TEXT,
            command=lambda: self.change_font_size(-1),
        )
        btn_zoom_out.pack(side="left", padx=1)

        self.lbl_font_size = ctk.CTkLabel(
            font_box,
            text=f"{self.font_size}pt",
            width=32,
            font=ctk.CTkFont(family="monospace", size=9),
            text_color=T.MUTED,
        )
        self.lbl_font_size.pack(side="left", padx=2)

        btn_zoom_in = ctk.CTkButton(
            font_box,
            text="A+",
            width=26,
            height=22,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=T.BORDER,
            hover_color=T.SURFACE,
            text_color=T.TEXT,
            command=lambda: self.change_font_size(1),
        )
        btn_zoom_in.pack(side="left", padx=1)

        # Autoscroll switch
        cb_auto = ctk.CTkCheckBox(
            controls,
            text="Autoscroll",
            variable=self.autoscroll,
            font=ctk.CTkFont(size=10),
            text_color=T.MUTED,
            fg_color=T.BORDER,
            checkmark_color=T.PRIMARY,
            checkbox_width=18,
            checkbox_height=18,
        )
        cb_auto.pack(side="right", padx=8)

    def _build_text_area(self):
        # We use standard tk.Text with dark configuration for full control over tag coloring
        self.textbox = tk.Text(
            self,
            bg=T.resolve(T.TERMINAL),
            fg=T.resolve(T.MUTED),
            insertbackground=T.resolve(T.BLUE),
            relief="flat",
            font=("Consolas", self.font_size),
            wrap="word",
        )
        self.scrollbar = ctk.CTkScrollbar(self, command=self.textbox.yview)
        self.textbox.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y", padx=(0, 6), pady=(0, 6))
        self.textbox.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=(0, 6))

        self._apply_text_theme()
        self._bind_zoom_shortcuts()

    def _set_appearance_mode(self, mode):
        super()._set_appearance_mode(mode)
        if hasattr(self, "textbox"):
            self._apply_text_theme()

    def _apply_text_theme(self):
        self.textbox.configure(bg=T.resolve(T.TERMINAL), fg=T.resolve(T.TEXT),
                               insertbackground=T.resolve(T.BLUE))
        for tag, color in {"info": T.BLUE, "metrics": T.SUCCESS, "round": T.WARNING,
                           "error": T.ERROR, "clients": T.PURPLE, "timestamp": T.MUTED}.items():
            self.textbox.tag_config(tag, foreground=T.resolve(color))

    def _bind_zoom_shortcuts(self):
        self.textbox.bind("<Control-MouseWheel>", self._on_ctrl_mousewheel)
        self.textbox.bind("<Control-plus>", lambda e: (self.change_font_size(1), "break")[1])
        self.textbox.bind("<Control-equal>", lambda e: (self.change_font_size(1), "break")[1])
        self.textbox.bind("<Control-minus>", lambda e: (self.change_font_size(-1), "break")[1])
        self.textbox.bind("<Control-0>", lambda e: (self.set_font_size(10), "break")[1])

    def _on_ctrl_mousewheel(self, event):
        if event.delta > 0:
            self.change_font_size(1)
        elif event.delta < 0:
            self.change_font_size(-1)
        return "break"

    def change_font_size(self, delta: int):
        """Increase or decrease console font size within [7, 24] pt."""
        self.set_font_size(self.font_size + delta)

    def set_font_size(self, size: int):
        """Set exact console font size clamped within [7, 24] pt."""
        self.font_size = max(7, min(24, size))
        self.textbox.configure(font=("Consolas", self.font_size))
        if hasattr(self, "lbl_font_size"):
            self.lbl_font_size.configure(text=f"{self.font_size}pt")

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
