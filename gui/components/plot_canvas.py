from gui import theme as T
import os
import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MaxNLocator
import numpy as np
from typing import List, Dict, Optional, Any, Callable, Tuple


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
        self._draw_after_id: Optional[str] = None
        self._resize_after_id: Optional[str] = None
        self._syncing_host: bool = False

        self._build_legend_ribbon()
        self.plot_host = ctk.CTkFrame(self, fg_color="transparent")
        self.plot_host.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.plot_host.pack_propagate(False)
        self.plot_host.bind("<Configure>", self._on_plot_host_configure)
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

        # Action buttons packed FIRST on Right so narrow screens never push them off
        btn_save = ctk.CTkButton(
            self.legend_frame,
            text="Save Plot",
            width=78,
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
            width=82,
            height=28,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=T.PRIMARY,
            hover_color=T.PRIMARY_HOVER,
            text_color=T.ON_ACCENT,
            command=self.export_csv,
        )
        btn_csv.pack(side="right", padx=4)

        # Dynamic container for Server vs Client specific controls
        self.ribbon_mode_frame = ctk.CTkFrame(self.legend_frame, fg_color="transparent")
        self.ribbon_mode_frame.pack(side="left", fill="both", expand=True)
        self.legend_frame.bind("<Configure>", self._on_ribbon_configure)

        self._populate_mode_ribbon()

    def _on_ribbon_configure(self, event=None):
        """Show or hide optional legend chips when ribbon width is tight."""
        try:
            w = int(event.width) if event is not None else int(self.legend_frame.winfo_width())
            if w <= 20:
                return
            logical_w = w / T.get_widget_scale(self.legend_frame)
            if self.mode == "server" and self.server_active_view == "global" and hasattr(self, "badge_box"):
                if logical_w < 820 and self.badge_box.winfo_manager():
                    self.badge_box.pack_forget()
                elif logical_w >= 820 and not self.badge_box.winfo_manager():
                    self.badge_box.pack(side="left", padx=4)
            elif self.mode == "client" and hasattr(self, "client_badge_box"):
                if logical_w < 760 and self.client_badge_box.winfo_manager():
                    self.client_badge_box.pack_forget()
                elif logical_w >= 760 and not self.client_badge_box.winfo_manager():
                    self.client_badge_box.pack(side="left", padx=4)
        except Exception:
            pass

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
            if self.server_active_view.startswith("client_"):
                try:
                    self._update_client_summary_label(int(self.server_active_view.split("_")[-1]))
                except Exception:
                    pass

            self.badge_box = ctk.CTkFrame(self.ribbon_mode_frame, fg_color="transparent")
            badges = [
                ("┄ BTA", T.ACCENT),
                ("┄ BMCTA", T.BLUE),
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

    def _update_client_summary_label(self, cid: int) -> None:
        """Update the ribbon summary text for the inspected server client."""
        if not hasattr(self, "lbl_client_summary") or not self.lbl_client_summary.winfo_exists():
            return
        pkg = self.client_packages.get(int(cid), {})
        history = pkg.get("history", []) or []
        eval_h = pkg.get("eval_history", []) or []

        samples = pkg.get("samples", 0) or 0
        if not samples and history:
            samples = history[-1].get("num_samples", 0) or 0

        eval_accs = [
            (float(e["eval_accuracy"]) * 100.0 if float(e["eval_accuracy"]) <= 1.0 else float(e["eval_accuracy"]))
            for e in eval_h
            if e.get("eval_accuracy") is not None
        ]
        if eval_accs:
            best_acc = max(eval_accs)
        else:
            train_accs = [
                (float(h["train_accuracy"]) * 100.0 if float(h["train_accuracy"]) <= 1.0 else float(h["train_accuracy"]))
                for h in history
                if h.get("train_accuracy") is not None
            ]
            best_acc = max(train_accs, default=0.0)

        self.lbl_client_summary.configure(
            text=f"Client {int(cid)} | Samples: {int(samples):,} | Best Acc: {best_acc:.1f}%"
        )

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
                self._update_client_summary_label(cid)
            except Exception:
                self.server_active_view = "global"

        self._build_figure()
        self._refresh_plots()

    @staticmethod
    def _merge_client_entries(
        existing: List[Dict[str, Any]],
        incoming: List[Dict[str, Any]],
        round_keys: Tuple[str, ...],
        canonical_key: str,
        require_any_of: Optional[Tuple[str, ...]] = None,
    ) -> List[Dict[str, Any]]:
        """Merge round-keyed dictionaries cleanly without clobbering prior rounds."""
        by_round: Dict[int, Dict[str, Any]] = {}
        for entry in list(existing or []) + list(incoming or []):
            if not isinstance(entry, dict):
                continue
            if require_any_of and all(entry.get(k) is None for k in require_any_of):
                continue
            r_val = None
            for rk in round_keys:
                if entry.get(rk) is not None:
                    try:
                        r_val = int(entry[rk])
                        break
                    except (TypeError, ValueError):
                        pass
            if r_val is None:
                continue
            merged = dict(by_round.get(r_val, {}))
            for k, v in entry.items():
                if v is not None:
                    merged[k] = v
            merged[canonical_key] = r_val
            by_round[r_val] = merged
        return [by_round[r] for r in sorted(by_round.keys())]

    def add_client_package(self, payload: Dict[str, Any]):
        """Merge a client's round or final metrics & distribution package and refresh active inspection view."""
        cid_raw = payload.get("client_id")
        if cid_raw is None:
            return
        cid = int(cid_raw)

        existed = cid in self.client_packages
        pkg = self.client_packages.setdefault(
            cid,
            {
                "client_id": cid,
                "samples": 0,
                "distribution": {},
                "history": [],
                "eval_history": [],
            },
        )
        prev_state = (
            pkg.get("samples", 0),
            dict(pkg.get("distribution") or {}),
            [dict(h) for h in (pkg.get("history") or [])],
            [dict(e) for e in (pkg.get("eval_history") or [])],
        )

        samples_val = payload.get("samples") if payload.get("samples") is not None else payload.get("num_samples")
        if samples_val is not None:
            try:
                if int(samples_val) > 0:
                    pkg["samples"] = int(samples_val)
            except (TypeError, ValueError):
                pass

        dist = payload.get("distribution")
        if isinstance(dist, dict) and dist:
            merged_dist = dict(pkg.get("distribution") or {})
            merged_dist.update(dist)
            pkg["distribution"] = merged_dist

        incoming_hist = list(payload.get("history") or [])
        flat_round = payload.get("server_round") if payload.get("server_round") is not None else payload.get("round")
        if flat_round is not None and (
            payload.get("train_loss") is not None or payload.get("train_accuracy") is not None
        ):
            flat_fit: Dict[str, Any] = {"round": int(flat_round)}
            if payload.get("train_loss") is not None:
                flat_fit["train_loss"] = payload["train_loss"]
            if payload.get("train_accuracy") is not None:
                flat_fit["train_accuracy"] = payload["train_accuracy"]
            if payload.get("num_samples") is not None:
                flat_fit["num_samples"] = payload["num_samples"]
            incoming_hist.append(flat_fit)

        pkg["history"] = self._merge_client_entries(
            pkg.get("history", []),
            incoming_hist,
            round_keys=("round", "server_round"),
            canonical_key="round",
        )

        if not pkg["samples"] and pkg["history"]:
            try:
                last_samples = pkg["history"][-1].get("num_samples", 0)
                if last_samples:
                    pkg["samples"] = int(last_samples)
            except (TypeError, ValueError):
                pass

        incoming_eval = list(payload.get("eval_history") or [])
        if flat_round is not None and (
            payload.get("eval_loss") is not None or payload.get("eval_accuracy") is not None
        ):
            flat_eval: Dict[str, Any] = {"server_round": int(flat_round)}
            if payload.get("eval_loss") is not None:
                flat_eval["eval_loss"] = payload["eval_loss"]
            if payload.get("eval_accuracy") is not None:
                flat_eval["eval_accuracy"] = payload["eval_accuracy"]
            incoming_eval.append(flat_eval)

        pkg["eval_history"] = self._merge_client_entries(
            pkg.get("eval_history", []),
            incoming_eval,
            round_keys=("server_round", "round"),
            canonical_key="server_round",
            require_any_of=("eval_loss", "eval_accuracy"),
        )
        pkg["rounds"] = max(len(pkg["history"]), len(pkg["eval_history"]))
        new_state = (
            pkg.get("samples", 0),
            dict(pkg.get("distribution") or {}),
            pkg.get("history") or [],
            pkg.get("eval_history") or [],
        )
        pkg_changed = (not existed) or (new_state != prev_state)

        if self.mode == "server" and hasattr(self, "cmb_server_view") and self.cmb_server_view.winfo_exists():
            c_options = ["Global Server Metrics"] + [f"Client {c}" for c in sorted(self.client_packages.keys())]
            try:
                existing_vals = list(self.cmb_server_view.cget("values") or [])
            except Exception:
                existing_vals = []
            if existing_vals != c_options:
                self.cmb_server_view.configure(values=c_options)

        if self.mode == "server" and self.server_active_view == f"client_{cid}":
            needs_fig = getattr(self, "ax_dist", None) is None
            if needs_fig:
                self._build_figure()
            if pkg_changed or needs_fig:
                self._update_client_summary_label(cid)
                self._refresh_plots()

    def _set_appearance_mode(self, mode):
        super()._set_appearance_mode(mode)
        if hasattr(self, "fig") and hasattr(self, "canvas"):
            self.fig.set_facecolor(T.resolve(T.PANEL))
            self._refresh_plots()

    def _compute_initial_figsize(self, default_w: float, default_h: float) -> Tuple[float, float]:
        """Compute initial figure size in inches matching the host container if already laid out."""
        host = getattr(self, "plot_host", None)
        if host is not None and host.winfo_exists():
            host_w = host.winfo_width()
            host_h = host.winfo_height()
            if host_w > 20 and host_h > 20:
                try:
                    import sys
                    if sys.platform == "win32":
                        ratio = round(float(host.tk.call("tk", "scaling")) / (96.0 / 72.0), 2)
                    else:
                        ratio = 1.0
                    ratio = max(1.0, ratio)
                except Exception:
                    ratio = 1.0
                eff_dpi = 100.0 * ratio
                return max(2.0, host_w / eff_dpi), max(1.2, host_h / eff_dpi)
        return default_w, default_h

    def _on_plot_host_configure(self, event=None):
        """Debounce figure resizing to match the plot host widget bounds across DPIs."""
        if self._syncing_host:
            return
        if getattr(self, "_resize_after_id", None) is not None:
            try:
                self.after_cancel(self._resize_after_id)
            except Exception:
                pass
            self._resize_after_id = None
        try:
            if self.winfo_exists():
                self._resize_after_id = self.after(15, self._sync_figure_to_host)
        except Exception:
            self._resize_after_id = None

    def _sync_figure_to_host(self):
        """Force Matplotlib Figure and TkAgg backing canvas to match plot_host pixel bounds."""
        self._resize_after_id = None
        if self._syncing_host:
            return
        host = getattr(self, "plot_host", None)
        canvas = getattr(self, "canvas", None)
        fig = getattr(self, "fig", None)
        if host is None or canvas is None or fig is None:
            return
        try:
            if not host.winfo_exists():
                return
            host_w = int(host.winfo_width())
            host_h = int(host.winfo_height())
        except Exception:
            return
        if host_w <= 20 or host_h <= 20:
            return

        dpival = float(getattr(fig, "dpi", 100.0) or 100.0)
        cur_w_px = fig.get_figwidth() * dpival
        cur_h_px = fig.get_figheight() * dpival
        try:
            photo_w = int(canvas._tkphoto.width())
            photo_h = int(canvas._tkphoto.height())
        except Exception:
            photo_w, photo_h = int(round(cur_w_px)), int(round(cur_h_px))

        if (
            abs(cur_w_px - host_w) <= 2
            and abs(cur_h_px - host_h) <= 2
            and abs(photo_w - host_w) <= 2
            and abs(photo_h - host_h) <= 2
        ):
            return

        self._syncing_host = True
        try:
            evt = type("_HostResizeEvent", (), {"width": host_w, "height": host_h})()
            canvas.resize(evt)
        except Exception:
            try:
                fig.set_size_inches(max(2.0, host_w / dpival), max(1.2, host_h / dpival), forward=False)
            except Exception:
                pass
        finally:
            self._syncing_host = False

    def _build_figure(self):
        """Create Matplotlib figure layout (Accuracy left, Loss right, optional Distribution far right)."""
        if getattr(self, "_draw_after_id", None) is not None:
            try:
                self.after_cancel(self._draw_after_id)
            except Exception:
                pass
            self._draw_after_id = None
        if getattr(self, "_resize_after_id", None) is not None:
            try:
                self.after_cancel(self._resize_after_id)
            except Exception:
                pass
            self._resize_after_id = None

        if hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.get_tk_widget().destroy()

        host = getattr(self, "plot_host", None) or self

        if self.mode == "server" and self.server_active_view.startswith("client_"):
            # 1x3 panels: Accuracy | Loss | Data Distribution
            fig_w, fig_h = self._compute_initial_figsize(11.5, 3.6)
            self.fig = Figure(figsize=(fig_w, fig_h), dpi=100, facecolor=T.resolve(T.PANEL), layout="constrained")
            gs = self.fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.0])
            self.ax_acc = self.fig.add_subplot(gs[0, 0])
            self.ax_loss = self.fig.add_subplot(gs[0, 1], sharex=self.ax_acc)
            self.ax_dist = self.fig.add_subplot(gs[0, 2])
            self._style_axis(self.ax_acc, "Client Accuracy (Train & TA)", "Server Round", "Accuracy (%)")
            self._style_axis(self.ax_loss, "Client Loss (Train vs Eval)", "Server Round", "Loss")
            self._style_axis(self.ax_dist, "Data Distribution", "Class Label", "Sample Count")
        else:
            # 1x2 panels: Accuracy (Left) and Loss (Right) sharing X-axis
            fig_w, fig_h = self._compute_initial_figsize(10.5, 3.6)
            self.fig = Figure(figsize=(fig_w, fig_h), dpi=100, facecolor=T.resolve(T.PANEL), layout="constrained")
            self.ax_acc = self.fig.add_subplot(1, 2, 1)
            self.ax_loss = self.fig.add_subplot(1, 2, 2, sharex=self.ax_acc)
            self.ax_dist = None

            if self.mode == "server":
                self._style_axis(self.ax_acc, "Accuracy Progression", "Communication Round", "Accuracy")
                self._style_axis(self.ax_loss, "Evaluation Loss Reduction", "Communication Round", "Loss")
            else:
                x_label = "Communication Round" if self.client_view_mode == "round" else "Epoch (Current Round)"
                loss_title = "Local Round Loss" if self.client_view_mode == "round" else "Current Round Epoch Loss"
                acc_title = "Local Round Accuracies" if self.client_view_mode == "round" else "Current Round Epoch Acc"
                self._style_axis(self.ax_acc, acc_title, x_label, "Accuracy")
                self._style_axis(self.ax_loss, loss_title, x_label, "Loss")

        self.canvas = FigureCanvasTkAgg(self.fig, master=host)
        tk_widget = self.canvas.get_tk_widget()
        if host is self.plot_host:
            tk_widget.place(x=0, y=0, relwidth=1.0, relheight=1.0)
        else:
            tk_widget.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        tk_widget.bind("<Map>", lambda _e: self.after_idle(self._sync_figure_to_host), add="+")
        self._sync_figure_to_host()

    def _style_axis(self, ax, title: str, xlabel: str, ylabel: str):
        host_h = self.plot_host.winfo_height() if getattr(self, "plot_host", None) is not None else 0
        compact = (host_h > 20 and host_h < 280)
        ax.set_facecolor(T.resolve(T.PANEL))
        ax.set_title(title, color=T.resolve(T.TEXT), fontsize=10 if compact else 10.5, fontweight="bold", pad=5)
        ax.set_xlabel(xlabel, color=T.resolve(T.MUTED), fontsize=8.5 if compact else 9.0)
        ax.set_ylabel(ylabel, color=T.resolve(T.MUTED), fontsize=8.5 if compact else 9.0)
        ax.grid(True, linestyle="--", alpha=0.6, color=T.resolve(T.BORDER))
        ax.tick_params(colors=T.resolve(T.MUTED), labelsize=8.0 if compact else 8.5)
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
        existed = r_num in self._server_round_map
        entry = self._server_round_map.setdefault(
            r_num, {"loss": None, "bta": None, "bmcta": None, "ta": None, "mcta": None}
        )
        changed = not existed
        for key, val in (("loss", loss), ("bta", bta), ("bmcta", bmcta), ("ta", ta), ("mcta", mcta)):
            if val is not None:
                f_val = float(val)
                if entry[key] != f_val:
                    entry[key] = f_val
                    changed = True

        if not changed:
            return

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
                            ta: Optional[float] = None, bta: Optional[float] = None,
                            eval_loss: Optional[float] = None):
        """Merge and plot client round metrics keyed by round across 1..R."""
        r_num = int(r)
        entry = self._client_round_map.setdefault(
            r_num, {"loss": None, "train_acc": None, "ta": None, "bta": None, "eval_loss": None}
        )
        if train_loss is not None:
            entry["loss"] = float(train_loss)
            entry["train_loss"] = float(train_loss)
        if eval_loss is not None:
            entry["eval_loss"] = float(eval_loss)
            if entry.get("train_loss") is None and entry["loss"] is None:
                entry["loss"] = float(eval_loss)
        if train_acc is not None:
            entry["train_acc"] = float(train_acc)
        if ta is not None:
            entry["ta"] = float(ta)
        if bta is not None:
            entry["bta"] = float(bta)
        elif ta is not None:
            prev_btas = [
                float(v["bta"])
                for rk, v in self._client_round_map.items()
                if rk <= r_num and v.get("bta") is not None
            ]
            entry["bta"] = max([float(ta)] + prev_btas)

        self._sync_client_round_arrays()

        if self.mode == "client" and self.client_view_mode == "round":
            self._refresh_plots()

    def _schedule_canvas_draw(self):
        """Coalesce Matplotlib canvas repaints via a short 25ms debounce timer."""
        if not hasattr(self, "canvas") or self.canvas is None:
            return
        if getattr(self, "_draw_after_id", None) is not None:
            return
        try:
            if self.winfo_exists():
                self._draw_after_id = self.after(25, self._flush_canvas_draw)
        except Exception:
            self._draw_after_id = None

    def _flush_canvas_draw(self):
        self._draw_after_id = None
        if not hasattr(self, "canvas") or self.canvas is None:
            return
        try:
            if self.winfo_exists():
                self._sync_figure_to_host()
                self.canvas.draw_idle()
        except Exception:
            pass

    def _refresh_plots(self):
        """Re-render data points on active Matplotlib axes and schedule coalesced canvas redraw."""
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

        self._schedule_canvas_draw()

    def _render_server_global(self):
        """Render global server curves (TA/MCTA lines + BTA/BMCTA horizontal dashed lines)."""
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

        all_accs: List[float] = []
        ta_pts = [(r, self._server_round_map[r]["ta"]) for r in sorted_rounds if self._server_round_map[r]["ta"] is not None]
        mcta_pts = [(r, self._server_round_map[r]["mcta"]) for r in sorted_rounds if self._server_round_map[r]["mcta"] is not None]
        bta_vals = [float(self._server_round_map[r]["bta"]) for r in sorted_rounds if self._server_round_map[r]["bta"] is not None]
        bmcta_vals = [float(self._server_round_map[r]["bmcta"]) for r in sorted_rounds if self._server_round_map[r]["bmcta"] is not None]

        # Plot TA as wider solid line underneath MCTA so overlapping TA==MCTA values remain distinct
        if ta_pts:
            xs_ta, ys_ta = zip(*ta_pts)
            self.ax_acc.plot(
                xs_ta,
                ys_ta,
                color=T.resolve(T.SUCCESS),
                lw=2.8,
                linestyle="-",
                marker="o",
                markersize=5.5,
                alpha=0.9,
                zorder=2,
                label=f"TA ({ys_ta[-1] * 100:.2f}%)",
            )
            all_accs.extend(ys_ta)

        if mcta_pts:
            xs_mcta, ys_mcta = zip(*mcta_pts)
            self.ax_acc.plot(
                xs_mcta,
                ys_mcta,
                color=T.resolve(T.PURPLE),
                lw=1.8,
                linestyle="--",
                marker="^",
                markersize=4.5,
                zorder=3,
                label=f"MCTA ({ys_mcta[-1] * 100:.2f}%)",
            )
            all_accs.extend(ys_mcta)

        # Horizontal dashed lines for overall best TA (with max(TA) fallback) and best MCTA
        bta_best = max(bta_vals) if bta_vals else (max(v for _, v in ta_pts) if ta_pts else None)
        if bta_best is not None:
            self.ax_acc.axhline(
                y=bta_best,
                color=T.resolve(T.ACCENT),
                lw=2.0,
                linestyle=(0, (5, 4)),
                zorder=4,
                label=f"BTA ({bta_best * 100:.2f}%)",
            )
            all_accs.append(bta_best)

        bmcta_best = max(bmcta_vals) if bmcta_vals else None
        if bmcta_best is not None:
            # Phase-shifted dash sequence keeps BTA and BMCTA alternating when values are equal
            self.ax_acc.axhline(
                y=bmcta_best,
                color=T.resolve(T.BLUE),
                lw=2.0,
                linestyle=(4.5, (5, 4)),
                zorder=5,
                label=f"BMCTA ({bmcta_best * 100:.2f}%)",
            )
            all_accs.append(bmcta_best)

        if all_accs and sorted_rounds:
            self._set_axis_limits(self.ax_acc, sorted_rounds, all_accs, is_acc=True)
            self.ax_acc.legend(
                facecolor=T.resolve(T.PANEL),
                edgecolor=T.resolve(T.BORDER),
                labelcolor=T.resolve(T.TEXT),
                fontsize=8.0,
                loc="lower right",
                ncol=2,
                columnspacing=1.0,
                handlelength=2.0,
                borderpad=0.35,
                framealpha=0.9,
            )

    def _render_server_client(self, cid: int):
        """Render picked client's round curves and data distribution bar chart."""
        self._style_axis(self.ax_loss, f"Client {cid} - Loss", "Server Round", "Loss")
        self._style_axis(self.ax_acc, f"Client {cid} - Accuracy", "Server Round", "Accuracy (%)")
        self.ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax_acc.xaxis.set_major_locator(MaxNLocator(integer=True))
        if self.ax_dist is not None:
            self._style_axis(self.ax_dist, f"Client {cid} - Class Distribution", "Class", "Samples")

        pkg = self.client_packages.get(cid, {})
        history = pkg.get("history", [])
        eval_history = pkg.get("eval_history", [])
        dist_dict = pkg.get("distribution", {})

        # 1. Loss Panel
        train_loss_pts = [
            (int(h.get("round", h.get("server_round", i + 1))), float(h["train_loss"]))
            for i, h in enumerate(history)
            if h.get("train_loss") is not None
        ]
        eval_loss_pts = [
            (int(e.get("server_round", e.get("round", i + 1))), float(e["eval_loss"]))
            for i, e in enumerate(eval_history)
            if e.get("eval_loss") is not None
        ]
        rounds_h = [r for r, _ in train_loss_pts]
        losses_h = [v for _, v in train_loss_pts]
        eval_rounds_l = [r for r, _ in eval_loss_pts]
        eval_losses = [v for _, v in eval_loss_pts]

        if rounds_h and losses_h:
            self.ax_loss.plot(rounds_h, losses_h, color=T.resolve(T.WARNING), lw=2, marker="o", markersize=3, label="Train Loss")
        if eval_rounds_l and eval_losses:
            self.ax_loss.plot(eval_rounds_l, eval_losses, color=T.resolve(T.ERROR), lw=2, marker="s", markersize=3, label="Test Loss")
        if losses_h or eval_losses:
            all_l = losses_h + eval_losses
            all_rounds_l = sorted(set(rounds_h) | set(eval_rounds_l))
            self._set_axis_limits(self.ax_loss, all_rounds_l, all_l)
            self.ax_loss.legend(facecolor=T.resolve(T.PANEL), edgecolor=T.resolve(T.BORDER), labelcolor=T.resolve(T.TEXT), fontsize=9)

        # 2. Accuracy Panel
        train_acc_pts = [
            (
                int(h.get("round", h.get("server_round", i + 1))),
                float(h["train_accuracy"]) * 100.0 if float(h["train_accuracy"]) <= 1.0 else float(h["train_accuracy"]),
            )
            for i, h in enumerate(history)
            if h.get("train_accuracy") is not None
        ]
        eval_acc_pts = [
            (
                int(e.get("server_round", e.get("round", i + 1))),
                float(e["eval_accuracy"]) * 100.0 if float(e["eval_accuracy"]) <= 1.0 else float(e["eval_accuracy"]),
            )
            for i, e in enumerate(eval_history)
            if e.get("eval_accuracy") is not None
        ]
        rounds_a = [r for r, _ in train_acc_pts]
        accs_h = [v for _, v in train_acc_pts]
        eval_rounds_a = [r for r, _ in eval_acc_pts]
        eval_accs = [v for _, v in eval_acc_pts]

        if rounds_a and accs_h:
            self.ax_acc.plot(rounds_a, accs_h, color=T.resolve(T.BLUE), lw=2, marker="o", markersize=3, label="Train Acc")
        if eval_rounds_a and eval_accs:
            self.ax_acc.plot(eval_rounds_a, eval_accs, color=T.resolve(T.ACCENT), lw=2, marker="s", markersize=3, label="Test TA")
        if accs_h or eval_accs:
            all_a = accs_h + eval_accs
            all_rounds_a = sorted(set(rounds_a) | set(eval_rounds_a))
            self._set_axis_limits(self.ax_acc, all_rounds_a, all_a, is_acc=True, max_val_100=True)
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

    def destroy(self):
        for attr in ("_draw_after_id", "_resize_after_id"):
            aid = getattr(self, attr, None)
            if aid is not None:
                try:
                    self.after_cancel(aid)
                except Exception:
                    pass
                setattr(self, attr, None)
        super().destroy()

    def clear_server(self):
        """Reset only server-side curves, round maps, and client telemetry packages."""
        self._server_round_map.clear()
        self._sync_server_arrays()
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

    def clear_client(self):
        """Reset only client-side curves (rounds and local epochs)."""
        self._client_round_map.clear()
        self._sync_client_round_arrays()
        self.client_epoch_data = {"epochs": [], "losses": [], "train_accs": []}

        self._build_figure()
        self._refresh_plots()

    def clear(self):
        """Reset all curves and round maps."""
        self.clear_server()
        self.clear_client()

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
