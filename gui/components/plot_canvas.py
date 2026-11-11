import os
import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import List, Dict, Optional


class DualMetricPlotCanvas(ctk.CTkFrame):
    """Dual-panel live Matplotlib canvas with an independent legend header ribbon to prevent any overlap."""

    def __init__(self, master, mode: str = "server", **kwargs):
        super().__init__(master, fg_color="#161920", corner_radius=8, **kwargs)
        self.mode = mode  # "server" or "client"

        # State storage for curves
        self.rounds: List[int] = []
        self.losses: List[float] = []
        self.btas: List[float] = []
        self.bmctas: List[float] = []
        self.tas: List[float] = []
        self.mctas: List[float] = []
        self.train_accs: List[float] = []

        self._build_legend_ribbon()
        self._build_figure()

    def _build_legend_ribbon(self):
        """Dedicated top legend frame outside plot axes to prevent text/line clipping."""
        self.legend_frame = ctk.CTkFrame(self, fg_color="#1e293b", height=32, corner_radius=6)
        self.legend_frame.pack(fill="x", padx=10, pady=(8, 4))

        title_text = "📈 SERVER LIVE METRICS" if self.mode == "server" else "📈 CLIENT LOCAL METRICS"
        title_lbl = ctk.CTkLabel(
            self.legend_frame,
            text=title_text,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#f1f5f9",
        )
        title_lbl.pack(side="left", padx=12)

        # Legend badges
        if self.mode == "server":
            badges = [
                ("● BTA", "#38bdf8", "Best Test Accuracy"),
                ("● BMCTA", "#34d399", "Best Mean Client Acc"),
                ("● TA", "#fbbf24", "Current Test Acc"),
                ("● MCTA", "#c084fc", "Mean Client Test Acc"),
                ("● Global Loss", "#f87171", "CrossEntropy Loss"),
            ]
        else:
            badges = [
                ("● Local Train Loss", "#f87171", "Local Training Loss"),
                ("● Train Acc", "#38bdf8", "Training Accuracy"),
                ("● Local Test TA", "#34d399", "Current Local Test Acc"),
                ("● Best TA (BTA)", "#fbbf24", "Best Local Test Acc"),
            ]

        for text, color, tooltip in badges:
            chip = ctk.CTkLabel(
                self.legend_frame,
                text=text,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=color,
            )
            chip.pack(side="left", padx=8)

        # Action button to export figure
        btn_save = ctk.CTkButton(
            self.legend_frame,
            text="📷 Save Plot",
            width=80,
            height=22,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#334155",
            hover_color="#475569",
            command=self.export_plot_image,
        )
        btn_save.pack(side="right", padx=8)

    def _build_figure(self):
        """Create dual-panel Matplotlib figure with dark styling and reserved margins."""
        self.fig = Figure(figsize=(7.5, 3.2), dpi=100, facecolor="#161920", constrained_layout=True)

        # Left Subplot: Loss
        self.ax_loss = self.fig.add_subplot(1, 2, 1)
        self.ax_loss.set_facecolor("#0f172a")
        self.ax_loss.set_title("Evaluation Loss" if self.mode == "server" else "Local Train Loss", color="#e2e8f0", fontsize=10, pad=6)
        self.ax_loss.set_xlabel("Communication Round" if self.mode == "server" else "Round / Step", color="#94a3b8", fontsize=8.5)
        self.ax_loss.set_ylabel("Loss", color="#94a3b8", fontsize=8.5)
        self.ax_loss.grid(True, linestyle="--", alpha=0.25, color="#334155")
        self.ax_loss.tick_params(colors="#94a3b8", labelsize=8)

        # Right Subplot: Accuracies
        self.ax_acc = self.fig.add_subplot(1, 2, 2)
        self.ax_acc.set_facecolor("#0f172a")
        self.ax_acc.set_title("Accuracy Progression" if self.mode == "server" else "Local Accuracies", color="#e2e8f0", fontsize=10, pad=6)
        self.ax_acc.set_xlabel("Communication Round" if self.mode == "server" else "Round / Step", color="#94a3b8", fontsize=8.5)
        self.ax_acc.set_ylabel("Accuracy", color="#94a3b8", fontsize=8.5)
        self.ax_acc.grid(True, linestyle="--", alpha=0.25, color="#334155")
        self.ax_acc.tick_params(colors="#94a3b8", labelsize=8)

        # Plot lines
        (self.line_loss,) = self.ax_loss.plot([], [], color="#f87171", lw=2)

        if self.mode == "server":
            (self.line_bta,) = self.ax_acc.plot([], [], color="#38bdf8", lw=2)
            (self.line_bmcta,) = self.ax_acc.plot([], [], color="#34d399", lw=2)
            (self.line_ta,) = self.ax_acc.plot([], [], color="#fbbf24", lw=1.2, linestyle=":")
            (self.line_mcta,) = self.ax_acc.plot([], [], color="#c084fc", lw=1.2, linestyle="--")
        else:
            (self.line_train_acc,) = self.ax_acc.plot([], [], color="#38bdf8", lw=2)
            (self.line_ta,) = self.ax_acc.plot([], [], color="#34d399", lw=2)
            (self.line_bta,) = self.ax_acc.plot([], [], color="#fbbf24", lw=1.5, linestyle=":")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def update_server_metrics(self, r: int, loss: Optional[float] = None, bta: Optional[float] = None,
                              bmcta: Optional[float] = None, ta: Optional[float] = None, mcta: Optional[float] = None):
        """Update live curves for server metrics."""
        self.rounds.append(r)
        if loss is not None:
            self.losses.append(loss)
        if bta is not None:
            self.btas.append(bta)
        if bmcta is not None:
            self.bmctas.append(bmcta)
        if ta is not None:
            self.tas.append(ta)
        if mcta is not None:
            self.mctas.append(mcta)

        self._refresh_plots()

    def update_client_metrics(self, r: int, train_loss: Optional[float] = None, train_acc: Optional[float] = None,
                              ta: Optional[float] = None, bta: Optional[float] = None):
        """Update live curves for client metrics."""
        self.rounds.append(r)
        if train_loss is not None:
            self.losses.append(train_loss)
        if train_acc is not None:
            self.train_accs.append(train_acc)
        if ta is not None:
            self.tas.append(ta)
        if bta is not None:
            self.btas.append(bta)

        self._refresh_plots()

    def _refresh_plots(self):
        """Re-render data points and adjust limits with headroom buffer."""
        # Loss plot
        if self.losses:
            x_loss = self.rounds[:len(self.losses)]
            self.line_loss.set_data(x_loss, self.losses)
            self.ax_loss.set_xlim(min(x_loss), max(x_loss) + 1 if len(x_loss) > 1 else 2)
            min_l, max_l = min(self.losses), max(self.losses)
            pad = max(0.05, (max_l - min_l) * 0.1)
            self.ax_loss.set_ylim(max(0.0, min_l - pad), max_l + pad)

        # Accuracies plot
        all_accs = []
        if self.mode == "server":
            if self.btas:
                x = self.rounds[:len(self.btas)]
                self.line_bta.set_data(x, self.btas)
                all_accs.extend(self.btas)
            if self.bmctas:
                x = self.rounds[:len(self.bmctas)]
                self.line_bmcta.set_data(x, self.bmctas)
                all_accs.extend(self.bmctas)
            if self.tas:
                x = self.rounds[:len(self.tas)]
                self.line_ta.set_data(x, self.tas)
                all_accs.extend(self.tas)
            if self.mctas:
                x = self.rounds[:len(self.mctas)]
                self.line_mcta.set_data(x, self.mctas)
                all_accs.extend(self.mctas)
        else:
            if self.train_accs:
                x = self.rounds[:len(self.train_accs)]
                self.line_train_acc.set_data(x, self.train_accs)
                all_accs.extend(self.train_accs)
            if self.tas:
                x = self.rounds[:len(self.tas)]
                self.line_ta.set_data(x, self.tas)
                all_accs.extend(self.tas)
            if self.btas:
                x = self.rounds[:len(self.btas)]
                self.line_bta.set_data(x, self.btas)
                all_accs.extend(self.btas)

        if all_accs:
            min_a, max_a = min(all_accs), max(all_accs)
            self.ax_acc.set_xlim(min(self.rounds), max(self.rounds) + 1 if len(self.rounds) > 1 else 2)
            # 8% headroom buffer to prevent collision with top title
            self.ax_acc.set_ylim(max(0.0, min_a * 0.95), min(1.05, max_a * 1.08))

        self.canvas.draw_idle()

    def clear(self):
        """Reset curves."""
        self.rounds.clear()
        self.losses.clear()
        self.btas.clear()
        self.bmctas.clear()
        self.tas.clear()
        self.mctas.clear()
        self.train_accs.clear()

        self.line_loss.set_data([], [])
        if self.mode == "server":
            self.line_bta.set_data([], [])
            self.line_bmcta.set_data([], [])
            self.line_ta.set_data([], [])
            self.line_mcta.set_data([], [])
        else:
            self.line_train_acc.set_data([], [])
            self.line_ta.set_data([], [])
            self.line_bta.set_data([], [])

        self.canvas.draw_idle()

    def export_plot_image(self):
        """Save plot snapshot to results or user folder."""
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("SVG Vector", "*.svg"), ("PDF Document", "*.pdf")],
            initialfile=f"fedmedai_{self.mode}_metrics.png",
        )
        if path:
            self.fig.savefig(path, dpi=200, facecolor=self.fig.get_facecolor(), bbox_inches="tight")
