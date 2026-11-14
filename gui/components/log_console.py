from gui import theme as T
import tkinter as tk
import customtkinter as ctk
from typing import Dict, Any, Optional, Callable


class LiveLogConsole(ctk.CTkFrame):
    """Real-time scrolling terminal output console with syntax highlighting and hardware telemetry."""

    def __init__(
        self,
        master,
        title: str = "REAL-TIME LOG CONSOLE",
        show_source_switcher: bool = False,
        on_source_change: Optional[Callable[[str], None]] = None,
        **kwargs,
    ):
        super().__init__(master, fg_color=T.SIDEBAR, corner_radius=12, border_width=1, border_color=T.BORDER, **kwargs)
        self.autoscroll = tk.BooleanVar(value=True)
        self.font_size = 11
        self.show_source_switcher = show_source_switcher
        self.on_source_change = on_source_change
        self.mono_font = T.MONO_FONT

        self._build_header(title)
        self._build_text_area()

    def _build_header(self, title: str):
        self.header_frame = ctk.CTkFrame(self, fg_color=T.SURFACE, height=46, corner_radius=6)
        self.header_frame.pack(fill="x", padx=8, pady=(8, 4))

        # Row 1: Left has Title and optional Source Switcher, Right has Controls (Clear, Zoom, Autoscroll)
        row1 = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        row1.pack(fill="x", padx=6, pady=(4, 2))

        lbl = ctk.CTkLabel(
            row1,
            text=f"{title}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=T.TEXT,
        )
        lbl.pack(side="left", padx=(4, 6))

        if self.show_source_switcher:
            self.seg_source = ctk.CTkSegmentedButton(
                row1,
                values=["Server", "Client"],
                height=26,
                font=ctk.CTkFont(size=11, weight="bold"),
                fg_color=T.PANEL,
                selected_color=T.PRIMARY,
                selected_hover_color=T.PRIMARY_HOVER,
                unselected_color=T.PANEL,
                unselected_hover_color=T.SURFACE,
                text_color=T.TEXT,
                command=self._on_source_switch,
            )
            self.seg_source.set("Server")
            self.seg_source.pack(side="left", padx=4)

            self.lbl_auto = ctk.CTkLabel(
                row1,
                text="[Auto-Detect Active]",
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=T.SUCCESS,
            )
            self.lbl_auto.pack(side="left", padx=6)

        # Right side controls: Clear, Zoom (A-, pt, A+), Autoscroll
        btn_clear = ctk.CTkButton(
            row1,
            text="Clear",
            width=54,
            height=24,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=T.BORDER,
            hover_color=T.SURFACE,
            text_color=T.TEXT,
            command=self.clear,
        )
        btn_clear.pack(side="right", padx=(4, 2))

        # Font zoom controls (A- / 11pt / A+)
        font_box = ctk.CTkFrame(row1, fg_color="transparent")
        font_box.pack(side="right", padx=(4, 4))

        btn_zoom_out = ctk.CTkButton(
            font_box,
            text="A-",
            width=26,
            height=24,
            font=ctk.CTkFont(size=11, weight="bold"),
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
            font=ctk.CTkFont(family=self.mono_font, size=10),
            text_color=T.MUTED,
        )
        self.lbl_font_size.pack(side="left", padx=1)

        btn_zoom_in = ctk.CTkButton(
            font_box,
            text="A+",
            width=26,
            height=24,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=T.BORDER,
            hover_color=T.SURFACE,
            text_color=T.TEXT,
            command=lambda: self.change_font_size(1),
        )
        btn_zoom_in.pack(side="left", padx=1)

        cb_auto = ctk.CTkCheckBox(
            row1,
            text="Autoscroll",
            variable=self.autoscroll,
            font=ctk.CTkFont(size=11),
            text_color=T.MUTED,
            fg_color=T.BORDER,
            checkmark_color=T.PRIMARY,
            checkbox_width=16,
            checkbox_height=16,
        )
        cb_auto.pack(side="right", padx=6)

        # Row 2: Hardware Telemetry row
        row2 = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        row2.pack(fill="x", padx=6, pady=(0, 3))

        self.lbl_telemetry = ctk.CTkLabel(
            row2,
            text="GPU: Scanning... | VRAM: 0.0 GB",
            font=ctk.CTkFont(family=self.mono_font, size=10, weight="bold"),
            text_color=T.ACCENT,
        )
        self.lbl_telemetry.pack(side="left", padx=(4, 0))

    def _on_source_switch(self, choice: str):
        if self.on_source_change:
            self.on_source_change(choice)

    def _build_text_area(self):
        # Dedicated container to isolate geometry changes when zooming font
        self.text_container = ctk.CTkFrame(self, fg_color="transparent")
        self.text_container.pack(fill="both", expand=True, padx=8, pady=(0, 6))

        # We use standard tk.Text with width=1, height=1 to guarantee zero geometry shifting across zoom/DPI
        self.textbox = tk.Text(
            self.text_container,
            bg=T.resolve(T.TERMINAL),
            fg=T.resolve(T.MUTED),
            insertbackground=T.resolve(T.BLUE),
            relief="flat",
            font=(self.mono_font, self.font_size),
            wrap="word",
            width=1,
            height=1,
        )
        self.scrollbar = ctk.CTkScrollbar(self.text_container, command=self.textbox.yview)
        self.textbox.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y", padx=(2, 0))
        self.textbox.pack(side="left", fill="both", expand=True)

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
        self.textbox.configure(font=(self.mono_font, self.font_size))
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
        """Update hardware telemetry badge with real-time GPU, CPU, and RAM stats."""
        cpu = float(info.get("cpu_percent", 0.0))
        ram_u = float(info.get("ram_used_gb", 0.0))
        ram_t = float(info.get("ram_total_gb", 0.0))
        ram_p = float(info.get("ram_percent", 0.0))

        if not info.get("has_gpu"):
            text = f"CPU: {cpu:.1f}% | RAM: {ram_u:.1f}/{ram_t:.1f} GB ({ram_p:.0f}%) · CPU Only Mode"
        else:
            name = info.get("gpu_name") or "GPU"
            if len(name) > 16:
                name = name[:14] + ".."
            gpu_u = float(info.get("gpu_util", 0.0))
            gpu_temp = info.get("gpu_temp")
            temp_str = f" · {gpu_temp}°C" if gpu_temp is not None else ""
            used = float(info.get("vram_used_gb", 0.0))
            total = float(info.get("vram_total_gb", 0.0))
            vram_p = float(info.get("vram_percent", 0.0))
            text = f"{name}: {gpu_u:.0f}%{temp_str} · VRAM: {used:.1f}/{total:.1f} GB ({vram_p:.0f}%) | CPU: {cpu:.1f}% | RAM: {ram_u:.1f}/{ram_t:.1f} GB"

        self.after(0, lambda: self.lbl_telemetry.configure(text=text))

    def clear(self):
        self.textbox.delete("1.0", tk.END)
