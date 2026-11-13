"""Native (light, dark) color pairs shared by the desktop dashboard."""

import customtkinter as ctk

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
