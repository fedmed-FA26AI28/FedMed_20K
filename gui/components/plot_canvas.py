from gui import theme as T
import os
import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MaxNLocator
import numpy as np
from typing import List, Dict, Optional, Any, Callable


class DualMetricPlotCanvas(ctk.CTkFrame):
    """Dual/Triple-panel live Matplotlib canvas with legend header ribbon, client inspection, and epoch reset."""

    def __init__(self, master, mode: str = "server", on_role_change: Optional[Callable[[str], None]] = None, **kwargs):
        super().__init__(master, fg_color=T.PANEL, corner_radius=12, border_width=1, border_color=T.BORDER, **kwargs)
        self.mode = mode  # "server" or "client"
        self.on_role_change = on_role_change

        # Server Global state storage keyed by round number (no duplicate rounds)
        self._server_round_map: Dict[int, Dict[str, Optional[float]]] = {}
        self.rounds: List[int] = []
        self.losses: List[float] = []
        self.btas: List[float] = []
        self.bmctas: List[float] = []
        self.tas: List[float] = []
        self.mctas: List[float] = []

        # Client state storage: Round vs Epoch separation
        self._client_round_map: Dict[int, Dict[str, Optional[float]]] = {}
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
        """Dedicated top legend frame outside plot axes with role switcher and interactive view controls."""
        self.legend_frame = ctk.CTkFrame(self, fg_color=T.SURFACE, height=38, corner_radius=6, border_width=1, border_color=T.BORDER)
        self.legend_frame.pack(fill="x", padx=10, pady=(8, 4))
        self.legend_frame.pack_propagate(False)

        self.seg_role = ctk.CTkSegmentedButton(
            self.legend_frame,
            values=["Server", "Client"],
            height=28,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=T.BORDER,
            selected_color=T.PRIMARY,
            selected_hover_color=T.PRIMARY_HOVER,
            unselected_color=T.PANEL,
            unselected_hover_color=T.SURFACE,
            text_color=T.TEXT,
            command=self._on_role_segmented_changed,
        )
        self.seg_role.set("Server" if self.mode == "server" else "Client")
        self.seg_role.pack(side="left", padx=(10, 6))

        # Dynamic container for Server vs Client specific controls
        self.ribbon_mode_frame = ctk.CTkFrame(self.legend_frame, fg_color="transparent")
        self.ribbon_mode_frame.pack(side="left", fill="both", expand=True)

        # Action buttons: Export CSV + Save Plot (always on Right)
        btn_save = ctk.CTkButton(
            self.legend_frame,
            text="Save Plot",
            width=88,
            height=28,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=T.PANEL,
            hover_color=T.SURFACE,
            text_color=T.TEXT,
            border_width=1,
            border_color=T.BORDER,
            command=self.export_plot_image,
        )
        btn_save.pack(side="right", padx=(4, 8))

        btn_csv = ctk.CTkButton(
            self.legend_frame,
            text="Export CSV",
            width=88,
            height=28,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=T.PRIMARY,
            hover_color=T.PRIMARY_HOVER,
            text_color=T.ON_ACCENT,
            command=self.export_csv,
        )
        btn_csv.pack(side="right", padx=4)

        self._populate_mode_ribbon()

    def _populate_mode_ribbon(self):
        """Populate Server or Client controls inside the ribbon container."""
        for w in self.ribbon_mode_frame.winfo_children():
            w.destroy()

        if self.mode == "server":
            self.lbl_server_title = ctk.CTkLabel(
                self.ribbon_mode_frame,
                text="SERVER METRICS:",
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=T.TEXT,
            )
            self.lbl_server_title.pack(side="left", padx=(4, 4))

            c_options = ["Global Server Metrics"] + [f"Client {c}" for c in sorted(self.client_packages.keys())]
            self.cmb_server_view = ctk.CTkComboBox(
                self.ribbon_mode_frame,
                values=c_options,
                width=190,
                height=28,
                font=ctk.CTkFont(size=11, weight="bold"),
                fg_color=T.PANEL,
                border_color=T.BORDER,
                text_color=T.TEXT,
                button_color=T.PRIMARY,
                button_hover_color=T.PRIMARY_HOVER,
                command=self._on_server_view_changed,
            )
            if self.server_active_view.startswith("client_"):
                cid_str = self.server_active_view.split("_")[-1]
                self.cmb_server_view.set(f"Client {cid_str}")
            else:
                self.cmb_server_view.set("Global Server Metrics")
            self.cmb_server_view.pack(side="left", padx=4)

            self.lbl_client_summary = ctk.CTkLabel(
                self.ribbon_mode_frame,
                text="",
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=T.ACCENT,
            )
            self.lbl_client_summary.pack(side="left", padx=6)

            self.badge_box = ctk.CTkFrame(self.ribbon_mode_frame, fg_color="transparent")
            badges = [
                ("● BTA", T.PRIMARY),
                ("● BMCTA", T.BLUE),
                ("● TA", T.SUCCESS),
                ("● MCTA", T.PURPLE),
                ("● Loss", T.WARNING),
            ]
            for text, color in badges:
                chip = ctk.CTkLabel(
                    self.badge_box,
                    text=text,
                    font=ctk.CTkFont(size=11, weight="bold"),
                    text_color=color,
                )
                chip.pack(side="left", padx=4)
            if self.server_active_view == "global":
                self.badge_box.pack(side="left", padx=6)
        else:
            title_lbl = ctk.CTkLabel(
                self.ribbon_mode_frame,
                text="CLIENT METRICS:",
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=T.TEXT,
            )
            title_lbl.pack(side="left", padx=(4, 4))

            self.seg_client_view = ctk.CTkSegmentedButton(
                self.ribbon_mode_frame,
                values=["Round Progress", "Current Epochs"],
                height=28,
                font=ctk.CTkFont(size=11, weight="bold"),
                fg_color=T.BORDER,
                selected_color=T.PRIMARY,
                selected_hover_color=T.PRIMARY_HOVER,
                unselected_color=T.PANEL,
                unselected_hover_color=T.SURFACE,
                text_color=T.TEXT,
                command=self._on_client_view_mode_changed,
            )
            self.seg_client_view.set("Round Progress" if self.client_view_mode == "round" else "Current Epochs")
            self.seg_client_view.pack(side="left", padx=6)

            self.client_badge_box = ctk.CTkFrame(self.ribbon_mode_frame, fg_color="transparent")
            self.client_badge_box.pack(side="left", padx=6)
            self._update_client_ribbon_badges()

    def _on_role_segmented_changed(self, choice: str):
        self.set_mode(choice.lower(), notify=True)

    def set_mode(self, mode: str, notify: bool = False):
        """Switch plot mode between 'server' and 'client' and rebuild controls/axes if changed."""
        target = "client" if mode.lower() == "client" else "server"
        self.seg_role.set("Client" if target == "client" else "Server")
        changed = (target != self.mode)
        self.mode = target
        if changed:
            self._populate_mode_ribbon()
            self._build_figure()
            self._refresh_plots()
        if notify and self.on_role_change:
            self.on_role_change(target)

    def _update_client_ribbon_badges(self):
        """Update chips on client ribbon based on active view mode."""
        for w in self.client_badge_box.winfo_children():
            w.destroy()

        if self.client_view_mode == "round":
            badges = [
                ("● Train Loss", T.WARNING),
                ("● Train Acc", T.BLUE),
                ("● Test TA", T.PRIMARY),
                ("● Best TA", T.SUCCESS),
            ]
        else:
            badges = [
                ("● Epoch Loss", T.WARNING),
                ("● Epoch Acc", T.PRIMARY),
                ("(Auto-resets each round)", T.MUTED),
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
                    text=f"Client {cid} | Samples: {samples:,} | Best Acc: {best_acc:.1f}%"
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
            if self.mode == "server" and hasattr(self, "cmb_server_view") and self.cmb_server_view.winfo_exists():
                c_options = ["Global Server Metrics"] + [f"Client {c}" for c in sorted(self.client_packages.keys())]
                self.cmb_server_view.configure(values=c_options)

    def _set_appearance_mode(self, mode):
        super()._set_appearance_mode(mode)
        if hasattr(self, "fig") and hasattr(self, "canvas"):
            self.fig.set_facecolor(T.resolve(T.PANEL))
            self._refresh_plots()

    def _build_figure(self):
        """Create Matplotlib side-by-side figure layout (Accuracy left, Loss right, optional Distribution far right)."""
        if hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.get_tk_widget().destroy()

        if self.mode == "server" and self.server_active_view.startswith("client_"):
            # 1x3 Side-by-Side: Accuracy | Loss | Data Distribution
            self.fig = Figure(figsize=(11.5, 3.8), dpi=100, facecolor=T.resolve(T.PANEL), constrained_layout=True)
            gs = self.fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.0])
            self.ax_acc = self.fig.add_subplot(gs[0, 0])
            self.ax_loss = self.fig.add_subplot(gs[0, 1], sharex=self.ax_acc)
            self.ax_dist = self.fig.add_subplot(gs[0, 2])
            self._style_axis(self.ax_acc, "Client Accuracy (Train & TA)", "Server Round", "Accuracy (%)")
            self._style_axis(self.ax_loss, "Client Loss (Train vs Eval)", "Server Round", "Loss")
            self._style_axis(self.ax_dist, "Data Distribution", "Class Label", "Sample Count")
        else:
            # 1x2 Side-by-Side Panels: Accuracy (Left) and Loss (Right) sharing X-axis
            self.fig = Figure(figsize=(10.5, 3.8), dpi=100, facecolor=T.resolve(T.PANEL), constrained_layout=True)
            self.ax_acc = self.fig.add_subplot(1, 2, 1)
            self.ax_loss = self.fig.add_subplot(1, 2, 2, sharex=self.ax_acc)
            self.ax_dist = None

            if self.mode == "server":
                self._style_axis(self.ax_acc, "Accuracy Progression (BTA / BMCTA)", "Communication Round", "Accuracy")
                self._style_axis(self.ax_loss, "Evaluation Loss Reduction", "Communication Round", "Loss")
            else:
                x_label = "Communication Round" if self.client_view_mode == "round" else "Epoch (Current Round)"
                loss_title = "Local Round Loss" if self.client_view_mode == "round" else "Current Round Epoch Loss"
                acc_title = "Local Round Accuracies" if self.client_view_mode == "round" else "Current Round Epoch Acc"
                self._style_axis(self.ax_acc, acc_title, x_label, "Accuracy")
                self._style_axis(self.ax_loss, loss_title, x_label, "Loss")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def _style_axis(self, ax, title: str, xlabel: str, ylabel: str):
        ax.set_facecolor(T.resolve(T.PANEL))
        ax.set_title(title, color=T.resolve(T.TEXT), fontsize=11, fontweight="bold", pad=6)
        ax.set_xlabel(xlabel, color=T.resolve(T.MUTED), fontsize=9.5)
        ax.set_ylabel(ylabel, color=T.resolve(T.MUTED), fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.6, color=T.resolve(T.BORDER))
        ax.tick_params(colors=T.resolve(T.MUTED), labelsize=9)
        if "bottom" in ax.spines:
            ax.spines["bottom"].set_color(T.resolve(T.MUTED))
        if "left" in ax.spines:
            ax.spines["left"].set_color(T.resolve(T.MUTED))
        if "top" in ax.spines:
            ax.spines["top"].set_visible(False)
        if "right" in ax.spines:
            ax.spines["right"].set_visible(False)

    def _sync_server_arrays(self):
        """Rebuild sorted de-duplicated server arrays from _server_round_map."""
        sorted_rounds = sorted(self._server_round_map.keys())
        self.rounds = sorted_rounds
        self.losses = [self._server_round_map[r]["loss"] for r in sorted_rounds if self._server_round_map[r]["loss"] is not None]
        self.btas = [self._server_round_map[r]["bta"] for r in sorted_rounds if self._server_round_map[r]["bta"] is not None]
        self.bmctas = [self._server_round_map[r]["bmcta"] for r in sorted_rounds if self._server_round_map[r]["bmcta"] is not None]
        self.tas = [self._server_round_map[r]["ta"] for r in sorted_rounds if self._server_round_map[r]["ta"] is not None]
        self.mctas = [self._server_round_map[r]["mcta"] for r in sorted_rounds if self._server_round_map[r]["mcta"] is not None]

    def _sync_client_round_arrays(self):
        """Rebuild sorted de-duplicated client round arrays from _client_round_map."""
        sorted_rounds = sorted(self._client_round_map.keys())
        self.client_round_data = {
            "rounds": sorted_rounds,
            "losses": [self._client_round_map[r]["loss"] for r in sorted_rounds if self._client_round_map[r]["loss"] is not None],
            "train_accs": [self._client_round_map[r]["train_acc"] for r in sorted_rounds if self._client_round_map[r]["train_acc"] is not None],
            "tas": [self._client_round_map[r]["ta"] for r in sorted_rounds if self._client_round_map[r]["ta"] is not None],
            "btas": [self._client_round_map[r]["bta"] for r in sorted_rounds if self._client_round_map[r]["bta"] is not None],
        }

    def update_server_metrics(self, r: int, loss: Optional[float] = None, bta: Optional[float] = None,
                              bmcta: Optional[float] = None, ta: Optional[float] = None, mcta: Optional[float] = None):
        """Merge and plot live curves for server global metrics keyed by round."""
        r_num = int(r)
        entry = self._server_round_map.setdefault(
            r_num, {"loss": None, "bta": None, "bmcta": None, "ta": None, "mcta": None}
        )
        if loss is not None:
            entry["loss"] = float(loss)
        if bta is not None:
            entry["bta"] = float(bta)
        if bmcta is not None:
            entry["bmcta"] = float(bmcta)
        if ta is not None:
            entry["ta"] = float(ta)
        if mcta is not None:
            entry["mcta"] = float(mcta)

        self._sync_server_arrays()

        if self.mode == "server" and self.server_active_view == "global":
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

        if self.mode == "client" and self.client_view_mode == "epoch":
            self._refresh_plots()

    def update_client_round(self, r: int, train_loss: Optional[float] = None, train_acc: Optional[float] = None,
                            ta: Optional[float] = None, bta: Optional[float] = None):
        """Merge and plot client round metrics keyed by round across 1..R."""
        r_num = int(r)
        entry = self._client_round_map.setdefault(
            r_num, {"loss": None, "train_acc": None, "ta": None, "bta": None}
        )
        if train_loss is not None:
            entry["loss"] = float(train_loss)
        if train_acc is not None:
            entry["train_acc"] = float(train_acc)
        if ta is not None:
            entry["ta"] = float(ta)
        if bta is not None:
            entry["bta"] = float(bta)

        self._sync_client_round_arrays()

        if self.mode == "client" and self.client_view_mode == "round":
            self._refresh_plots()

    def _refresh_plots(self):
        """Re-render data points on active Matplotlib axes and force immediate canvas redraw."""
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

        if hasattr(self, "canvas") and self.canvas is not None:
            try:
                self.canvas.draw()
            except Exception:
                self.canvas.draw_idle()

    def _render_server_global(self):
        """Render global server metrics curves from consolidated round map."""
        self._style_axis(self.ax_loss, "Evaluation Loss", "Communication Round", "Loss")
        self._style_axis(self.ax_acc, "Accuracy Progression", "Communication Round", "Accuracy")
        self.ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax_acc.xaxis.set_major_locator(MaxNLocator(integer=True))

        sorted_rounds = sorted(self._server_round_map.keys())
        loss_pts = [(r, self._server_round_map[r]["loss"]) for r in sorted_rounds if self._server_round_map[r]["loss"] is not None]
        if loss_pts:
            xs, ys = zip(*loss_pts)
            self.ax_loss.plot(xs, ys, color=T.resolve(T.WARNING), lw=2, marker="o", markersize=4)
            self._set_axis_limits(self.ax_loss, list(xs), list(ys))

        all_accs = []
        acc_specs = [
            ("bta", T.ACCENT, 2.0, "-", "BTA"),
            ("bmcta", T.BLUE, 2.0, "-", "BMCTA"),
            ("ta", T.SUCCESS, 1.5, ":", "TA"),
            ("mcta", T.PURPLE, 1.5, "--", "MCTA"),
        ]
        for key, token, lw, ls, label in acc_specs:
            pts = [(r, self._server_round_map[r][key]) for r in sorted_rounds if self._server_round_map[r][key] is not None]
            if pts:
                xs, ys = zip(*pts)
                self.ax_acc.plot(xs, ys, color=T.resolve(token), lw=lw, linestyle=ls, marker="o", markersize=4, label=label)
                all_accs.extend(ys)

        if all_accs and sorted_rounds:
            self._set_axis_limits(self.ax_acc, sorted_rounds, all_accs, is_acc=True)

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
            self.ax_loss.plot(rounds_h, losses_h, color=T.resolve(T.WARNING), lw=2, marker="o", markersize=3, label="Train Loss")
        eval_rounds = [e.get("server_round", i + 1) for i, e in enumerate(eval_history)]
        eval_losses = [e.get("eval_loss", 0.0) for e in eval_history if e.get("eval_loss") is not None]
        if eval_rounds and eval_losses:
            self.ax_loss.plot(eval_rounds[:len(eval_losses)], eval_losses, color=T.resolve(T.ERROR), lw=2, marker="s", markersize=3, label="Test Loss")
        if losses_h or eval_losses:
            all_l = losses_h + eval_losses
            self._set_axis_limits(self.ax_loss, rounds_h or eval_rounds, all_l)
            self.ax_loss.legend(facecolor=T.resolve(T.PANEL), edgecolor=T.resolve(T.BORDER), labelcolor=T.resolve(T.TEXT), fontsize=9)

        # 2. Accuracy Panel
        accs_h = [h.get("train_accuracy", 0.0) * 100 for h in history]
        if rounds_h and accs_h:
            self.ax_acc.plot(rounds_h, accs_h, color=T.resolve(T.BLUE), lw=2, marker="o", markersize=3, label="Train Acc")
        eval_accs = [e.get("eval_accuracy", 0.0) * 100 for e in eval_history]
        if eval_rounds and eval_accs:
            self.ax_acc.plot(eval_rounds, eval_accs, color=T.resolve(T.ACCENT), lw=2, marker="s", markersize=3, label="Test TA")
        if accs_h or eval_accs:
            all_a = accs_h + eval_accs
            self._set_axis_limits(self.ax_acc, rounds_h or eval_rounds, all_a, is_acc=True, max_val_100=True)
            self.ax_acc.legend(facecolor=T.resolve(T.PANEL), edgecolor=T.resolve(T.BORDER), labelcolor=T.resolve(T.TEXT), fontsize=9)

        # 3. Data Distribution Panel
        if self.ax_dist is not None and dist_dict:
            try:
                classes = sorted(dist_dict.keys(), key=lambda x: int(x))
            except Exception:
                classes = sorted(dist_dict.keys())
            counts = [int(dist_dict[c]) for c in classes]
            x_labels = [f"C{c}" for c in classes]

            base_colors = [T.resolve(T.PRIMARY), T.resolve(T.BLUE), T.resolve(T.PURPLE), T.resolve(T.WARNING), T.resolve(T.ERROR), T.resolve(T.SUCCESS), T.resolve(T.BLUE), T.resolve(T.WARNING_ACTION)]
            bar_colors = [base_colors[i % len(base_colors)] for i in range(len(classes))]

            bars = self.ax_dist.bar(x_labels, counts, color=bar_colors, alpha=0.85, edgecolor=T.resolve(T.MUTED))
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
                    fontsize=9,
                    fontweight="bold",
                    color=T.resolve(T.TEXT),
                )

    def _render_client_round(self):
        """Render client cumulative round curves from consolidated round map."""
        self._style_axis(self.ax_loss, "Local Round Loss", "Communication Round", "Loss")
        self._style_axis(self.ax_acc, "Local Round Accuracies", "Communication Round", "Accuracy")
        self.ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax_acc.xaxis.set_major_locator(MaxNLocator(integer=True))

        sorted_rounds = sorted(self._client_round_map.keys())
        loss_pts = [(r, self._client_round_map[r]["loss"]) for r in sorted_rounds if self._client_round_map[r]["loss"] is not None]
        if loss_pts:
            xs, ys = zip(*loss_pts)
            self.ax_loss.plot(xs, ys, color=T.resolve(T.WARNING), lw=2, marker="o", markersize=4)
            self._set_axis_limits(self.ax_loss, list(xs), list(ys))

        all_accs = []
        acc_specs = [
            ("train_acc", T.BLUE, 2.0, "-", "Train Acc"),
            ("ta", T.ACCENT, 2.0, "-", "TA"),
            ("bta", T.SUCCESS, 1.5, ":", "Best TA"),
        ]
        for key, token, lw, ls, label in acc_specs:
            pts = [(r, self._client_round_map[r][key]) for r in sorted_rounds if self._client_round_map[r][key] is not None]
            if pts:
                xs, ys = zip(*pts)
                self.ax_acc.plot(xs, ys, color=T.resolve(token), lw=lw, linestyle=ls, marker="o", markersize=4, label=label)
                all_accs.extend(ys)

        if all_accs and sorted_rounds:
            self._set_axis_limits(self.ax_acc, sorted_rounds, all_accs, is_acc=True)

    def _render_client_epoch(self):
        """Render client active round epoch curves (resets each round)."""
        self._style_axis(self.ax_loss, "Current Round Epoch Loss", "Epoch", "Loss")
        self._style_axis(self.ax_acc, "Current Round Epoch Accuracy", "Epoch", "Accuracy")

        e_data = self.client_epoch_data
        epochs = e_data["epochs"]
        if epochs and e_data["losses"]:
            self.ax_loss.plot(epochs, e_data["losses"], color=T.resolve(T.WARNING), lw=2, marker="o", markersize=4)
            self._set_axis_limits(self.ax_loss, epochs, e_data["losses"])

        if epochs and e_data["train_accs"]:
            self.ax_acc.plot(epochs, e_data["train_accs"], color=T.resolve(T.ACCENT), lw=2, marker="o", markersize=4)
            self._set_axis_limits(self.ax_acc, epochs, e_data["train_accs"], is_acc=True)

    def _set_axis_limits(self, ax, x_vals, y_vals, is_acc: bool = False, max_val_100: bool = False):
        if not x_vals or not y_vals:
            return
        min_x, max_x = min(x_vals), max(x_vals)
        if min_x == max_x:
            ax.set_xlim(max(0.0, min_x - 0.5), max_x + 0.5)
        else:
            ax.set_xlim(min_x - 0.15, max_x + 0.5)
        min_y, max_y = min(y_vals), max(y_vals)
        if is_acc:
            top_bound = 105.0 if max_val_100 else 1.05
            ax.set_ylim(max(0.0, min_y * 0.92), min(top_bound, max_y * 1.08 + (1.0 if max_val_100 else 0.02)))
        else:
            pad = max(0.05, (max_y - min_y) * 0.1)
            ax.set_ylim(max(0.0, min_y - pad), max_y + pad)

    def clear(self):
        """Reset curves and round maps."""
        self._server_round_map.clear()
        self._client_round_map.clear()
        self._sync_server_arrays()
        self._sync_client_round_arrays()
        self.client_epoch_data = {"epochs": [], "losses": [], "train_accs": []}
        self.client_packages.clear()
        self.server_active_view = "global"

        if self.mode == "server" and hasattr(self, "cmb_server_view") and self.cmb_server_view.winfo_exists():
            self.cmb_server_view.configure(values=["Global Server Metrics"])
            self.cmb_server_view.set("Global Server Metrics")
            if hasattr(self, "lbl_client_summary") and self.lbl_client_summary.winfo_exists():
                self.lbl_client_summary.configure(text="")
            if hasattr(self, "badge_box") and self.badge_box.winfo_exists():
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

    def export_csv(self, filepath: Optional[str] = None, show_dialog: bool = True):
        """Export plotted curve points to CSV."""
        import csv
        from tkinter import filedialog, messagebox
        if self.mode == "server":
            if not self._server_round_map:
                if show_dialog:
                    messagebox.showinfo("Export CSV", "No server curve data to export yet.")
                return
            if not filepath:
                filepath = filedialog.asksaveasfilename(
                    title="Export Plotted Server Metrics CSV",
                    initialdir="results",
                    initialfile="server_plotted_metrics.csv",
                    filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                    defaultextension=".csv",
                )
                if not filepath:
                    return
            try:
                os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
                sorted_rounds = sorted(self._server_round_map.keys())
                with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    writer.writerow(["round", "loss", "bta", "bmcta", "ta", "mcta"])
                    for r in sorted_rounds:
                        rec = self._server_round_map[r]
                        writer.writerow([
                            r,
                            rec["loss"] if rec["loss"] is not None else "",
                            rec["bta"] if rec["bta"] is not None else "",
                            rec["bmcta"] if rec["bmcta"] is not None else "",
                            rec["ta"] if rec["ta"] is not None else "",
                            rec["mcta"] if rec["mcta"] is not None else "",
                        ])
                if show_dialog:
                    messagebox.showinfo("Export Complete", f"Exported {len(sorted_rounds)} rounds to:\n{filepath}")
            except Exception as e:
                print(f"[EXPORT CSV ERROR] {e}")
                if show_dialog:
                    messagebox.showerror("Export Error", f"Failed to export CSV:\n{e}")
        else:
            if self.client_view_mode == "round":
                if not self._client_round_map:
                    if show_dialog:
                        messagebox.showinfo("Export CSV", "No client curve data to export yet.")
                    return
                count = len(self._client_round_map)
            else:
                epochs = self.client_epoch_data.get("epochs") or []
                if not epochs:
                    if show_dialog:
                        messagebox.showinfo("Export CSV", "No client curve data to export yet.")
                    return
                count = len(epochs)
            if not filepath:
                filepath = filedialog.asksaveasfilename(
                    title="Export Plotted Client Metrics CSV",
                    initialdir="results",
                    initialfile=f"client_plotted_{self.client_view_mode}_metrics.csv",
                    filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                    defaultextension=".csv",
                )
                if not filepath:
                    return
            try:
                os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
                with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    if self.client_view_mode == "round":
                        writer.writerow(["round", "train_loss", "train_acc", "ta", "bta"])
                        for r in sorted(self._client_round_map.keys()):
                            rec = self._client_round_map[r]
                            writer.writerow([
                                r,
                                rec["loss"] if rec["loss"] is not None else "",
                                rec["train_acc"] if rec["train_acc"] is not None else "",
                                rec["ta"] if rec["ta"] is not None else "",
                                rec["bta"] if rec["bta"] is not None else "",
                            ])
                    else:
                        data = self.client_epoch_data
                        writer.writerow(["epoch", "train_loss", "train_acc"])
                        for i, ep in enumerate(data["epochs"]):
                            l = data["losses"][i] if i < len(data["losses"]) else ""
                            acc = data["train_accs"][i] if i < len(data["train_accs"]) else ""
                            writer.writerow([ep, l, acc])
                if show_dialog:
                    messagebox.showinfo("Export Complete", f"Exported {count} steps to:\n{filepath}")
            except Exception as e:
                print(f"[EXPORT CSV ERROR] {e}")
                if show_dialog:
                    messagebox.showerror("Export Error", f"Failed to export CSV:\n{e}")
