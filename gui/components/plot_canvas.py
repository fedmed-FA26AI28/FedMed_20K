import os
import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from typing import List, Dict, Optional, Any


class DualMetricPlotCanvas(ctk.CTkFrame):
    """Dual/Triple-panel live Matplotlib canvas with legend header ribbon, client inspection, and epoch reset."""

    def __init__(self, master, mode: str = "server", **kwargs):
        super().__init__(master, fg_color="#ffffff", corner_radius=8, border_width=1, border_color="#e8edf2", **kwargs)
        self.mode = mode  # "server" or "client"

        # Server Global state storage
        self.rounds: List[int] = []
        self.losses: List[float] = []
        self.btas: List[float] = []
        self.bmctas: List[float] = []
        self.tas: List[float] = []
        self.mctas: List[float] = []

        # Client state storage: Round vs Epoch separation
        self.client_round_data = {
            "rounds": [],
            "losses": [],
            "train_accs": [],
            "tas": [],
            "btas": [],
        }
        self.client_epoch_data = {
            "epochs": [],
            "losses": [],
            "train_accs": [],
        }
        self.client_view_mode = "round"  # "round" or "epoch"

        # Server: Client Inspection Packages (accumulated post-run / final round)
        self.client_packages: Dict[int, Dict[str, Any]] = {}
        self.server_active_view: str = "global"  # "global" or "client_<id>"

        self._build_legend_ribbon()
        self._build_figure()

    def _build_legend_ribbon(self):
        """Dedicated top legend frame outside plot axes with interactive view controls."""
        self.legend_frame = ctk.CTkFrame(self, fg_color="#f8fafc", height=34, corner_radius=6, border_width=1, border_color="#e8edf2")
        self.legend_frame.pack(fill="x", padx=10, pady=(8, 4))

        if self.mode == "server":
            # Server Mode: Dropdown to pick between Global view and individual clients
            self.lbl_server_title = ctk.CTkLabel(
                self.legend_frame,
                text="📈 SERVER:",
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color="#162d43",
            )
            self.lbl_server_title.pack(side="left", padx=(10, 4))

            self.cmb_server_view = ctk.CTkComboBox(
                self.legend_frame,
                values=["🌍 Global Server Metrics"],
                width=190,
                height=24,
                font=ctk.CTkFont(size=10, weight="bold"),
                fg_color="#ffffff",
                border_color="#dce3eb",
                text_color="#162d43",
                button_color="#0c8c80",
                button_hover_color="#087b70",
                command=self._on_server_view_changed,
            )
            self.cmb_server_view.pack(side="left", padx=4)

            self.lbl_client_summary = ctk.CTkLabel(
                self.legend_frame,
                text="",
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color="#0c8c80",
            )
            self.lbl_client_summary.pack(side="left", padx=8)

            # Global badges container (hidden when viewing a specific client)
            self.badge_box = ctk.CTkFrame(self.legend_frame, fg_color="transparent")
            self.badge_box.pack(side="left", padx=6)
            badges = [
                ("● BTA", "#0c8c80"),
                ("● BMCTA", "#3972b8"),
                ("● TA", "#0b8a5a"),
                ("● MCTA", "#8070c8"),
                ("● Loss", "#d79338"),
            ]
            for text, color in badges:
                chip = ctk.CTkLabel(
                    self.badge_box,
                    text=text,
                    font=ctk.CTkFont(size=10, weight="bold"),
                    text_color=color,
                )
                chip.pack(side="left", padx=4)

        else:
            # Client Mode: Segmented View Switcher between Round-Level and Current Epochs
            title_lbl = ctk.CTkLabel(
                self.legend_frame,
                text="📈 CLIENT:",
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color="#162d43",
            )
            title_lbl.pack(side="left", padx=(10, 4))

            self.seg_client_view = ctk.CTkSegmentedButton(
                self.legend_frame,
                values=["🔁 Round Progress", "⚡ Current Epochs"],
                height=24,
                font=ctk.CTkFont(size=10, weight="bold"),
                fg_color="#e2e8f0",
                selected_color="#0c8c80",
                selected_hover_color="#087b70",
                unselected_color="#ffffff",
                unselected_hover_color="#f1f5f9",
                text_color="#162d43",
                command=self._on_client_view_mode_changed,
            )
            self.seg_client_view.set("🔁 Round Progress")
            self.seg_client_view.pack(side="left", padx=6)

            self.client_badge_box = ctk.CTkFrame(self.legend_frame, fg_color="transparent")
            self.client_badge_box.pack(side="left", padx=6)
            self._update_client_ribbon_badges()

        # Action button to export figure
        btn_save = ctk.CTkButton(
            self.legend_frame,
            text="📷 Save Plot",
            width=80,
            height=22,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#ffffff",
            hover_color="#f3f7fa",
            text_color="#42566b",
            border_width=1,
            border_color="#dfe6ed",
            command=self.export_plot_image,
        )
        btn_save.pack(side="right", padx=8)

    def _update_client_ribbon_badges(self):
        """Update chips on client ribbon based on active view mode."""
        for w in self.client_badge_box.winfo_children():
            w.destroy()

        if self.client_view_mode == "round":
            badges = [
                ("● Train Loss", "#d79338"),
                ("● Train Acc", "#3972b8"),
                ("● Test TA", "#0c8c80"),
                ("● Best TA", "#0b8a5a"),
            ]
        else:
            badges = [
                ("● Epoch Loss", "#d79338"),
                ("● Epoch Acc", "#0c8c80"),
                ("(Auto-resets each round)", "#78889b"),
            ]
        for text, color in badges:
            chip = ctk.CTkLabel(
                self.client_badge_box,
                text=text,
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color=color,
            )
            chip.pack(side="left", padx=4)

    def _on_client_view_mode_changed(self, choice: str):
        self.client_view_mode = "round" if "Round" in choice else "epoch"
        self._update_client_ribbon_badges()
        self._build_figure()
        self._refresh_plots()

    def _on_server_view_changed(self, choice: str):
        if "Global" in choice:
            self.server_active_view = "global"
            self.lbl_client_summary.configure(text="")
            self.badge_box.pack(side="left", padx=6)
        else:
            try:
                cid = int(choice.split()[-1])
                self.server_active_view = f"client_{cid}"
                self.badge_box.pack_forget()
                pkg = self.client_packages.get(cid, {})
                samples = pkg.get("samples", 0)
                eval_h = pkg.get("eval_history", [])
                best_acc = max([e.get("eval_accuracy", 0.0) for e in eval_h], default=0.0) * 100
                self.lbl_client_summary.configure(
                    text=f"👤 Client {cid} | Samples: {samples:,} | Best Acc: {best_acc:.1f}%"
                )
            except Exception:
                self.server_active_view = "global"

        self._build_figure()
        self._refresh_plots()

    def add_client_package(self, payload: Dict[str, Any]):
        """Register a client's final metrics & data distribution package for post-training inspection."""
        cid = payload.get("client_id")
        if cid is not None:
            self.client_packages[int(cid)] = payload
            c_options = ["🌍 Global Server Metrics"] + [f"👤 Client {c}" for c in sorted(self.client_packages.keys())]
            self.cmb_server_view.configure(values=c_options)

    def _build_figure(self):
        """Create Matplotlib figure layout (2 panels for global/client, 3 panels for client inspection)."""
        if hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.get_tk_widget().destroy()

        if self.mode == "server" and self.server_active_view.startswith("client_"):
            # 3 Panels: Loss, Accuracy, and Data Distribution
            self.fig = Figure(figsize=(8.8, 3.2), dpi=100, facecolor="#ffffff", constrained_layout=True)
            self.ax_loss = self.fig.add_subplot(1, 3, 1)
            self.ax_acc = self.fig.add_subplot(1, 3, 2)
            self.ax_dist = self.fig.add_subplot(1, 3, 3)
            self._style_axis(self.ax_loss, "Client Loss (Train vs Eval)", "Server Round", "Loss")
            self._style_axis(self.ax_acc, "Client Accuracy (Train & TA)", "Server Round", "Accuracy (%)")
            self._style_axis(self.ax_dist, "Data Distribution", "Class Label", "Sample Count")
        else:
            # Standard 2 Panels
            self.fig = Figure(figsize=(7.5, 3.2), dpi=100, facecolor="#ffffff", constrained_layout=True)
            self.ax_loss = self.fig.add_subplot(1, 2, 1)
            self.ax_acc = self.fig.add_subplot(1, 2, 2)
            self.ax_dist = None

            if self.mode == "server":
                self._style_axis(self.ax_loss, "Evaluation Loss", "Communication Round", "Loss")
                self._style_axis(self.ax_acc, "Accuracy Progression", "Communication Round", "Accuracy")
            else:
                x_label = "Communication Round" if self.client_view_mode == "round" else "Epoch (Current Round)"
                loss_title = "Local Round Loss" if self.client_view_mode == "round" else "Current Round Epoch Loss"
                acc_title = "Local Round Accuracies" if self.client_view_mode == "round" else "Current Round Epoch Acc"
                self._style_axis(self.ax_loss, loss_title, x_label, "Loss")
                self._style_axis(self.ax_acc, acc_title, x_label, "Accuracy")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def _style_axis(self, ax, title: str, xlabel: str, ylabel: str):
        ax.set_facecolor("#fcfdfe")
        ax.set_title(title, color="#162d43", fontsize=9.5, fontweight="bold", pad=5)
        ax.set_xlabel(xlabel, color="#78889b", fontsize=8)
        ax.set_ylabel(ylabel, color="#78889b", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.6, color="#edf2f7")
        ax.tick_params(colors="#64748b", labelsize=7.5)
        if "bottom" in ax.spines:
            ax.spines["bottom"].set_color("#cbd5e1")
        if "left" in ax.spines:
            ax.spines["left"].set_color("#cbd5e1")
        if "top" in ax.spines:
            ax.spines["top"].set_visible(False)
        if "right" in ax.spines:
            ax.spines["right"].set_visible(False)

    def update_server_metrics(self, r: int, loss: Optional[float] = None, bta: Optional[float] = None,
                              bmcta: Optional[float] = None, ta: Optional[float] = None, mcta: Optional[float] = None):
        """Update live curves for server global metrics."""
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

        if self.server_active_view == "global":
            self._refresh_plots()

    def update_client_epoch(self, epoch: int, train_loss: float, train_acc: float, is_new_round: bool = False):
        """Update client epoch metrics; automatically resets at epoch 1 / new round to prevent zigzag."""
        if is_new_round or epoch == 1:
            self.client_epoch_data["epochs"].clear()
            self.client_epoch_data["losses"].clear()
            self.client_epoch_data["train_accs"].clear()

        self.client_epoch_data["epochs"].append(epoch)
        self.client_epoch_data["losses"].append(train_loss)
        self.client_epoch_data["train_accs"].append(train_acc)

        if self.client_view_mode == "epoch":
            self._refresh_plots()

    def update_client_round(self, r: int, train_loss: Optional[float] = None, train_acc: Optional[float] = None,
                            ta: Optional[float] = None, bta: Optional[float] = None):
        """Update client round metrics across rounds 1..R."""
        self.client_round_data["rounds"].append(r)
        if train_loss is not None:
            self.client_round_data["losses"].append(train_loss)
        if train_acc is not None:
            self.client_round_data["train_accs"].append(train_acc)
        if ta is not None:
            self.client_round_data["tas"].append(ta)
        if bta is not None:
            self.client_round_data["btas"].append(bta)

        if self.client_view_mode == "round":
            self._refresh_plots()

    def _refresh_plots(self):
        """Re-render data points on active Matplotlib axes."""
        if not hasattr(self, "ax_loss") or self.ax_loss is None:
            return

        self.ax_loss.cla()
        self.ax_acc.cla()
        if hasattr(self, "ax_dist") and self.ax_dist is not None:
            self.ax_dist.cla()

        if self.mode == "server":
            if self.server_active_view == "global":
                self._render_server_global()
            else:
                cid = int(self.server_active_view.split("_")[-1])
                self._render_server_client(cid)
        else:
            if self.client_view_mode == "round":
                self._render_client_round()
            else:
                self._render_client_epoch()

        self.canvas.draw_idle()

    def _render_server_global(self):
        """Render global server metrics curves."""
        self._style_axis(self.ax_loss, "Evaluation Loss", "Communication Round", "Loss")
        self._style_axis(self.ax_acc, "Accuracy Progression", "Communication Round", "Accuracy")

        if self.losses and self.rounds:
            x_l = self.rounds[:len(self.losses)]
            self.ax_loss.plot(x_l, self.losses, color="#d79338", lw=2, marker="o", markersize=3)
            self._set_axis_limits(self.ax_loss, x_l, self.losses)

        all_accs = []
        if self.rounds:
            if self.btas:
                x = self.rounds[:len(self.btas)]
                self.ax_acc.plot(x, self.btas, color="#0c8c80", lw=2, label="BTA")
                all_accs.extend(self.btas)
            if self.bmctas:
                x = self.rounds[:len(self.bmctas)]
                self.ax_acc.plot(x, self.bmctas, color="#3972b8", lw=2, label="BMCTA")
                all_accs.extend(self.bmctas)
            if self.tas:
                x = self.rounds[:len(self.tas)]
                self.ax_acc.plot(x, self.tas, color="#0b8a5a", lw=1.2, linestyle=":", label="TA")
                all_accs.extend(self.tas)
            if self.mctas:
                x = self.rounds[:len(self.mctas)]
                self.ax_acc.plot(x, self.mctas, color="#8070c8", lw=1.2, linestyle="--", label="MCTA")
                all_accs.extend(self.mctas)
            if all_accs:
                self._set_axis_limits(self.ax_acc, self.rounds, all_accs, is_acc=True)

    def _render_server_client(self, cid: int):
        """Render picked client's round curves and data distribution bar chart."""
        self._style_axis(self.ax_loss, f"Client {cid} - Loss", "Server Round", "Loss")
        self._style_axis(self.ax_acc, f"Client {cid} - Accuracy", "Server Round", "Accuracy (%)")
        if self.ax_dist is not None:
            self._style_axis(self.ax_dist, f"Client {cid} - Class Distribution", "Class", "Samples")

        pkg = self.client_packages.get(cid, {})
        history = pkg.get("history", [])
        eval_history = pkg.get("eval_history", [])
        dist_dict = pkg.get("distribution", {})

        # 1. Loss Panel
        rounds_h = [h.get("round", i + 1) for i, h in enumerate(history)]
        losses_h = [h.get("train_loss", 0.0) for h in history]
        if rounds_h and losses_h:
            self.ax_loss.plot(rounds_h, losses_h, color="#d79338", lw=2, marker="o", markersize=3, label="Train Loss")
        eval_rounds = [e.get("server_round", i + 1) for i, e in enumerate(eval_history)]
        eval_losses = [e.get("eval_loss", 0.0) for e in eval_history if e.get("eval_loss") is not None]
        if eval_rounds and eval_losses:
            self.ax_loss.plot(eval_rounds[:len(eval_losses)], eval_losses, color="#de7373", lw=2, marker="s", markersize=3, label="Test Loss")
        if losses_h or eval_losses:
            all_l = losses_h + eval_losses
            self._set_axis_limits(self.ax_loss, rounds_h or eval_rounds, all_l)
            self.ax_loss.legend(facecolor="#ffffff", edgecolor="#e2e8f0", labelcolor="#162d43", fontsize=7.5)

        # 2. Accuracy Panel
        accs_h = [h.get("train_accuracy", 0.0) * 100 for h in history]
        if rounds_h and accs_h:
            self.ax_acc.plot(rounds_h, accs_h, color="#3972b8", lw=2, marker="o", markersize=3, label="Train Acc")
        eval_accs = [e.get("eval_accuracy", 0.0) * 100 for e in eval_history]
        if eval_rounds and eval_accs:
            self.ax_acc.plot(eval_rounds, eval_accs, color="#0c8c80", lw=2, marker="s", markersize=3, label="Test TA")
        if accs_h or eval_accs:
            all_a = accs_h + eval_accs
            self._set_axis_limits(self.ax_acc, rounds_h or eval_rounds, all_a, is_acc=True, max_val_100=True)
            self.ax_acc.legend(facecolor="#ffffff", edgecolor="#e2e8f0", labelcolor="#162d43", fontsize=7.5)

        # 3. Data Distribution Panel
        if self.ax_dist is not None and dist_dict:
            try:
                classes = sorted(dist_dict.keys(), key=lambda x: int(x))
            except Exception:
                classes = sorted(dist_dict.keys())
            counts = [int(dist_dict[c]) for c in classes]
            x_labels = [f"C{c}" for c in classes]

            base_colors = ["#0c8c80", "#3972b8", "#8070c8", "#d79338", "#de7373", "#0b8a5a", "#2563eb", "#d97706"]
            bar_colors = [base_colors[i % len(base_colors)] for i in range(len(classes))]

            bars = self.ax_dist.bar(x_labels, counts, color=bar_colors, alpha=0.85, edgecolor="#cbd5e1")
            max_c = max(counts) if counts else 1
            self.ax_dist.set_ylim(0, max_c * 1.18 + 1)

            for bar, c_val in zip(bars, counts):
                h = bar.get_height()
                self.ax_dist.annotate(
                    f"{c_val}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                    color="#162d43",
                )

    def _render_client_round(self):
        """Render client cumulative round curves."""
        self._style_axis(self.ax_loss, "Local Round Loss", "Communication Round", "Loss")
        self._style_axis(self.ax_acc, "Local Round Accuracies", "Communication Round", "Accuracy")

        r_data = self.client_round_data
        rounds = r_data["rounds"]
        if rounds and r_data["losses"]:
            x = rounds[:len(r_data["losses"])]
            self.ax_loss.plot(x, r_data["losses"], color="#d79338", lw=2, marker="o", markersize=3)
            self._set_axis_limits(self.ax_loss, x, r_data["losses"])

        all_accs = []
        if rounds:
            if r_data["train_accs"]:
                x = rounds[:len(r_data["train_accs"])]
                self.ax_acc.plot(x, r_data["train_accs"], color="#3972b8", lw=2, label="Train Acc")
                all_accs.extend(r_data["train_accs"])
            if r_data["tas"]:
                x = rounds[:len(r_data["tas"])]
                self.ax_acc.plot(x, r_data["tas"], color="#0c8c80", lw=2, label="Local TA")
                all_accs.extend(r_data["tas"])
            if r_data["btas"]:
                x = rounds[:len(r_data["btas"])]
                self.ax_acc.plot(x, r_data["btas"], color="#0b8a5a", lw=1.5, linestyle=":", label="Best TA")
                all_accs.extend(r_data["btas"])
            if all_accs:
                self._set_axis_limits(self.ax_acc, rounds, all_accs, is_acc=True)

    def _render_client_epoch(self):
        """Render client active round epoch curves (resets each round)."""
        self._style_axis(self.ax_loss, "Current Round Epoch Loss", "Epoch", "Loss")
        self._style_axis(self.ax_acc, "Current Round Epoch Accuracy", "Epoch", "Accuracy")

        e_data = self.client_epoch_data
        epochs = e_data["epochs"]
        if epochs and e_data["losses"]:
            self.ax_loss.plot(epochs, e_data["losses"], color="#d79338", lw=2, marker="o", markersize=4)
            self._set_axis_limits(self.ax_loss, epochs, e_data["losses"])

        if epochs and e_data["train_accs"]:
            self.ax_acc.plot(epochs, e_data["train_accs"], color="#0c8c80", lw=2, marker="o", markersize=4)
            self._set_axis_limits(self.ax_acc, epochs, e_data["train_accs"], is_acc=True)

    def _set_axis_limits(self, ax, x_vals, y_vals, is_acc: bool = False, max_val_100: bool = False):
        if not x_vals or not y_vals:
            return
        ax.set_xlim(min(x_vals), max(x_vals) + 1 if len(x_vals) > 1 else max(x_vals) + 1)
        min_y, max_y = min(y_vals), max(y_vals)
        if is_acc:
            top_bound = 105.0 if max_val_100 else 1.05
            ax.set_ylim(max(0.0, min_y * 0.92), min(top_bound, max_y * 1.08 + (1.0 if max_val_100 else 0.02)))
        else:
            pad = max(0.05, (max_y - min_y) * 0.1)
            ax.set_ylim(max(0.0, min_y - pad), max_y + pad)

    def clear(self):
        """Reset curves."""
        self.rounds.clear()
        self.losses.clear()
        self.btas.clear()
        self.bmctas.clear()
        self.tas.clear()
        self.mctas.clear()

        self.client_round_data = {"rounds": [], "losses": [], "train_accs": [], "tas": [], "btas": []}
        self.client_epoch_data = {"epochs": [], "losses": [], "train_accs": []}
        self.client_packages.clear()

        if self.mode == "server":
            self.cmb_server_view.configure(values=["🌍 Global Server Metrics"])
            self.cmb_server_view.set("🌍 Global Server Metrics")
            self.server_active_view = "global"
            self.lbl_client_summary.configure(text="")
            self.badge_box.pack(side="left", padx=6)

        self._build_figure()
        self._refresh_plots()

    def export_plot_image(self):
        """Save plot snapshot to results or user folder."""
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("SVG Vector", "*.svg"), ("PDF Document", "*.pdf")],
            initialfile=f"fedmedai_{self.mode}_{self.server_active_view if self.mode == 'server' else self.client_view_mode}.png",
        )
        if path:
            self.fig.savefig(path, dpi=200, facecolor=self.fig.get_facecolor(), bbox_inches="tight")
