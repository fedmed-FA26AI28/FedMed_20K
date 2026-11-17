"""Native (light, dark) color pairs shared by the desktop dashboard."""

import platform
import customtkinter as ctk

# Cross-platform font family resolution
SYSTEM_OS = platform.system()
if SYSTEM_OS == "Windows":
    MONO_FONT = "Consolas"
    UI_FONT = "Segoe UI"
elif SYSTEM_OS == "Darwin":
    MONO_FONT = "Menlo"
    UI_FONT = "SF Pro Text"
else:
    MONO_FONT = "DejaVu Sans Mono"
    UI_FONT = "DejaVu Sans"

BACKGROUND = ("#f1f5f9", "#0b1220")
SIDEBAR = ("#e8eef5", "#0f172a")
PANEL = ("#ffffff", "#1e293b")
SURFACE = ("#f8fafc", "#162235")
BORDER = ("#cbd5e1", "#334155")
TEXT = ("#17263c", "#e2e8f0")
MUTED = ("#52647b", "#94a3b8")
ACCENT = ("#0f766e", "#14b8a6")
PRIMARY = ("#0f766e", "#0f766e")
PRIMARY_HOVER = ("#115e59", "#115e59")
ON_ACCENT = ("#ffffff", "#ffffff")
SELECTED = ("#ccfbf1", "#134e4a")
SELECTED_TEXT = ("#115e59", "#99f6e4")
BLUE = ("#0369a1", "#38bdf8")
PURPLE = ("#7c3aed", "#a78bfa")
SUCCESS = ("#047857", "#34d399")
WARNING = ("#92400e", "#fbbf24")
ERROR = ("#b91c1c", "#f87171")
DANGER = ("#b91c1c", "#b91c1c")
DANGER_HOVER = ("#991b1b", "#991b1b")
TERMINAL = ("#f8fafc", "#080f1d")

WARNING_ACTION = ("#92400e", "#92400e")
WARNING_HOVER = ("#78350f", "#78350f")
SUCCESS_SURFACE = ("#d1fae5", "#064e3b")
ERROR_SURFACE = ("#fee2e2", "#7f1d1d")


def resolve(color):
    """Resolve a native theme pair for Tk and Matplotlib, which require one color."""
    if isinstance(color, (tuple, list)):
        return color[0 if ctk.get_appearance_mode() == "Light" else 1]
    return color


def get_window_scale(win) -> float:
    """Return active CustomTkinter window DPI scaling factor (>= 1.0)."""
    try:
        getter = getattr(win, "_get_window_scaling", None)
        if callable(getter):
            return max(1.0, float(getter()))
    except Exception:
        pass
    return 1.0


def get_widget_scale(widget) -> float:
    """Return active CustomTkinter widget DPI scaling factor (>= 1.0)."""
    try:
        getter = getattr(widget, "_get_widget_scaling", None)
        if callable(getter):
            return max(1.0, float(getter()))
    except Exception:
        pass
    return 1.0


def fit_window_to_screen(
    win,
    target_w: int,
    target_h: int,
    min_w: int = None,
    min_h: int = None,
    parent=None,
):
    """Size and center a CTk/CTkToplevel window in logical units bounded by physical screen size."""
    scale = get_window_scale(win)
    try:
        screen_w = int(win.winfo_screenwidth())
        screen_h = int(win.winfo_screenheight())
    except Exception:
        screen_w, screen_h = 1920, 1080

    logical_w = screen_w / scale
    logical_h = screen_h / scale

    w = min(int(target_w), max(360, int(logical_w - 40)))
    h = min(int(target_h), max(300, int(logical_h - 64)))

    phys_w = round(w * scale)
    phys_h = round(h * scale)

    x = max(0, (screen_w - phys_w) // 2)
    y = max(0, (screen_h - phys_h) // 2)

    if parent is not None:
        try:
            if parent.winfo_exists() and parent.winfo_width() > 20 and parent.winfo_height() > 20:
                px = parent.winfo_rootx()
                py = parent.winfo_rooty()
                pw = parent.winfo_width()
                ph = parent.winfo_height()
                x = px + (pw - phys_w) // 2
                y = py + (ph - phys_h) // 2
                x = max(0, min(x, max(0, screen_w - phys_w)))
                y = max(0, min(y, max(0, screen_h - phys_h)))
        except Exception:
            pass

    win.geometry(f"{w}x{h}+{x}+{y}")
    if min_w is not None and min_h is not None:
        win.minsize(min(int(min_w), w), min(int(min_h), h))
    return w, h
