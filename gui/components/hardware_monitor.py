import threading
import tkinter as tk
import customtkinter as ctk
from typing import Dict, Any, Optional
from gui import theme as T


class HardwareMonitorWidget(ctk.CTkFrame):
    """Real-time hardware utilization monitor with animated gauges for CPU, RAM, and GPU/VRAM."""

    def __init__(self, master, **kwargs):
        super().__init__(
            master,
            fg_color=T.SURFACE,
            corner_radius=8,
            border_width=1,
            border_color=T.BORDER,
            **kwargs,
        )
        self._build_ui()

    def _build_ui(self):
        # 1. Header with Live Pulse Indicator
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=8, pady=(6, 4))

        lbl_title = ctk.CTkLabel(
            header,
            text="HARDWARE MONITOR",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=T.TEXT,
        )
        lbl_title.pack(side="left")

        self.lbl_status = ctk.CTkLabel(
            header,
            text="LIVE",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color=T.SUCCESS,
        )
        self.lbl_status.pack(side="right")

        # Divider
        div = ctk.CTkFrame(self, height=1, fg_color=T.BORDER)
        div.pack(fill="x", padx=8, pady=(0, 4))

        # 2. CPU Section
        cpu_row = ctk.CTkFrame(self, fg_color="transparent")
        cpu_row.pack(fill="x", padx=8, pady=(1, 0))

        lbl_cpu = ctk.CTkLabel(
            cpu_row,
            text="CPU",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=T.MUTED,
        )
        lbl_cpu.pack(side="left")

        self.lbl_cpu_val = ctk.CTkLabel(
            cpu_row,
            text="0.0%",
            font=ctk.CTkFont(family=T.MONO_FONT, size=10, weight="bold"),
            text_color=T.TEXT,
        )
        self.lbl_cpu_val.pack(side="right")

        self.bar_cpu = ctk.CTkProgressBar(
            self,
            height=6,
            corner_radius=3,
            fg_color=T.BORDER,
            progress_color=T.PRIMARY,
        )
        self.bar_cpu.set(0.0)
        self.bar_cpu.pack(fill="x", padx=8, pady=(1, 4))

        # 3. RAM Section
        ram_row = ctk.CTkFrame(self, fg_color="transparent")
        ram_row.pack(fill="x", padx=8, pady=(1, 0))

        lbl_ram = ctk.CTkLabel(
            ram_row,
            text="RAM",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=T.MUTED,
        )
        lbl_ram.pack(side="left")

        self.lbl_ram_val = ctk.CTkLabel(
            ram_row,
            text="0.0 / 0.0 GB",
            font=ctk.CTkFont(family=T.MONO_FONT, size=10, weight="bold"),
            text_color=T.TEXT,
        )
        self.lbl_ram_val.pack(side="right")

        self.bar_ram = ctk.CTkProgressBar(
            self,
            height=6,
            corner_radius=3,
            fg_color=T.BORDER,
            progress_color=T.BLUE,
        )
        self.bar_ram.set(0.0)
        self.bar_ram.pack(fill="x", padx=8, pady=(1, 4))

        # 4. GPU & VRAM Section
        gpu_row = ctk.CTkFrame(self, fg_color="transparent")
        gpu_row.pack(fill="x", padx=8, pady=(1, 0))

        self.lbl_gpu_title = ctk.CTkLabel(
            gpu_row,
            text="GPU",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=T.MUTED,
        )
        self.lbl_gpu_title.pack(side="left")

        self.lbl_gpu_val = ctk.CTkLabel(
            gpu_row,
            text="Scanning...",
            font=ctk.CTkFont(family=T.MONO_FONT, size=10, weight="bold"),
            text_color=T.ACCENT,
        )
        self.lbl_gpu_val.pack(side="right")

        self.bar_gpu = ctk.CTkProgressBar(
            self,
            height=6,
            corner_radius=3,
            fg_color=T.BORDER,
            progress_color=T.ACCENT,
        )
        self.bar_gpu.set(0.0)
        self.bar_gpu.pack(fill="x", padx=8, pady=(1, 2))

        # VRAM sub-stats
        vram_row = ctk.CTkFrame(self, fg_color="transparent")
        vram_row.pack(fill="x", padx=8, pady=(0, 0))

        self.lbl_vram_title = ctk.CTkLabel(
            vram_row,
            text="VRAM",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color=T.MUTED,
        )
        self.lbl_vram_title.pack(side="left")

        self.lbl_vram_val = ctk.CTkLabel(
            vram_row,
            text="0.0 / 0.0 GB",
            font=ctk.CTkFont(family=T.MONO_FONT, size=9, weight="bold"),
            text_color=T.PURPLE,
        )
        self.lbl_vram_val.pack(side="right")

        self.bar_vram = ctk.CTkProgressBar(
            self,
            height=5,
            corner_radius=2,
            fg_color=T.BORDER,
            progress_color=T.PURPLE,
        )
        self.bar_vram.set(0.0)
        self.bar_vram.pack(fill="x", padx=8, pady=(1, 6))

    @staticmethod
    def _cfg_if_changed(widget, **kwargs):
        cache = getattr(widget, "_last_cfg", None)
        if cache is None:
            cache = {}
            setattr(widget, "_last_cfg", cache)
        changed = {k: v for k, v in kwargs.items() if k not in cache or cache[k] != v}
        if changed:
            widget.configure(**changed)
            cache.update(changed)

    @staticmethod
    def _set_bar_if_changed(bar, val: float):
        clamped = min(1.0, max(0.0, float(val)))
        prev = getattr(bar, "_last_val", None)
        if prev is None or abs(clamped - prev) >= 0.002:
            bar.set(clamped)
            setattr(bar, "_last_val", clamped)

    def update_metrics(self, info: Dict[str, Any]):
        """Thread-safe update of hardware telemetry gauges."""
        if threading.current_thread() is threading.main_thread():
            self._apply_metrics(info)
        else:
            try:
                self.after(0, self._apply_metrics, info)
            except tk.TclError:
                pass

    def _apply_metrics(self, info: Dict[str, Any]):
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return

        cfg = self._cfg_if_changed
        set_bar = self._set_bar_if_changed

        # 1. Update CPU
        cpu_pct = float(info.get("cpu_percent", 0.0))
        cfg(self.lbl_cpu_val, text=f"{cpu_pct:.1f}%")
        set_bar(self.bar_cpu, cpu_pct / 100.0)
        cpu_color = T.ERROR if cpu_pct >= 90.0 else (T.WARNING if cpu_pct >= 75.0 else T.PRIMARY)
        cfg(self.bar_cpu, progress_color=cpu_color)

        # 2. Update RAM
        ram_used = float(info.get("ram_used_gb", 0.0))
        ram_total = float(info.get("ram_total_gb", 0.0))
        ram_pct = float(info.get("ram_percent", 0.0))
        cfg(self.lbl_ram_val, text=f"{ram_used:.1f}/{ram_total:.1f} GB")
        set_bar(self.bar_ram, ram_pct / 100.0)
        ram_color = T.ERROR if ram_pct >= 90.0 else (T.WARNING if ram_pct >= 80.0 else T.BLUE)
        cfg(self.bar_ram, progress_color=ram_color)

        # 3. Update GPU & VRAM
        has_gpu = info.get("has_gpu", False)
        if has_gpu:
            name = info.get("gpu_name") or "GPU"
            if len(name) > 14:
                name = name[:12] + ".."
            cfg(self.lbl_gpu_title, text=name)

            gpu_util = float(info.get("gpu_util", 0.0))
            gpu_temp = info.get("gpu_temp")
            temp_str = f" · {gpu_temp}°C" if gpu_temp is not None else ""
            cfg(self.lbl_gpu_val, text=f"{gpu_util:.0f}%{temp_str}", text_color=T.ACCENT)
            set_bar(self.bar_gpu, gpu_util / 100.0)

            vram_used = float(info.get("vram_used_gb", 0.0))
            vram_total = float(info.get("vram_total_gb", 0.0))
            vram_pct = float(info.get("vram_percent", 0.0))
            cfg(self.lbl_vram_val, text=f"{vram_used:.1f}/{vram_total:.1f} GB ({vram_pct:.0f}%)", text_color=T.PURPLE)
            set_bar(self.bar_vram, vram_pct / 100.0)

            vram_color = T.ERROR if vram_pct >= 90.0 else (T.WARNING if vram_pct >= 75.0 else T.PURPLE)
            cfg(self.bar_vram, progress_color=vram_color)
        else:
            cfg(self.lbl_gpu_title, text="GPU Mode")
            cfg(self.lbl_gpu_val, text="CPU Only", text_color=T.MUTED)
            set_bar(self.bar_gpu, 0.0)
            cfg(self.lbl_vram_val, text="No CUDA", text_color=T.MUTED)
            set_bar(self.bar_vram, 0.0)
