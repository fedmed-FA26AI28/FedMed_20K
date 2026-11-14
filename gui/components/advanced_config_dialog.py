"""
Advanced Configuration Modal Dialog.

Provides a professional multi-card sub-window allowing users to inspect,
fine-tune, and persist deep algorithmic hyperparameters, convergence policies,
evaluation routines, hardware data loaders, and partition path overrides.
"""

from __future__ import annotations

from gui import theme as T

import copy
import tkinter as tk
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import customtkinter as ctk

from gui.config_manager import PROJECT_ROOT

DEFAULT_SERVER_ADVANCED: Dict[str, Any] = {
    # 1. Strategy Mathematical Hyperparameters
    "proximal_mu": "0.01",
    "fedsol_rho": "2.0",
    "fedsol_target": "head",
    "fedsol_loss_type": "kl",
    "fedsol_temperature": "3.0",
    "fedsol_adaptive": False,
    "switch_round": "0",
    "fedswitch_weighting_mode": "bounded_sum",
    "fedswitch_coverage": "0.9",
    "fedswitch_epsilon": "1.0",
    "fedswitch_c_bound": "50",
    "scr_gamma": "1.0",
    "scr_head_lambda": "0.01",
    "fedsld_epsilon": "1e-6",
    "fedsld_reduction": "mean",
    # 2. Optimization Policies & Learning Rates
    "lr_mode": "fixed",
    "server_lr_decay_step": "10",
    "server_lr_decay_gamma": "0.5",
    "server_lr_min": "1e-5",
    "client_lr_patience": "2",
    "client_lr_factor": "0.5",
    "client_lr_min": "1e-6",
    "momentum": "0.9",
    "weight_decay": "0.0001",
    "clip_grad_norm": "",
    # 3. Evaluation & Early Stopping
    "server_eval": False,
    "local_test": True,
    "early_stop_patience": "0",
    "early_stop_metric": "accuracy",
    # 4. Data Partitions, Hardware & System
    "partition_path": "",
    "save_dir": "",
    "pin_memory": True,
    "num_workers": "0",
}

DEFAULT_CLIENT_ADVANCED: Dict[str, Any] = {
    # 1. Strategy Mathematical Hyperparameters
    "proximal_mu": "0.01",
    "fedsol_rho": "2.0",
    "fedsol_target": "head",
    "fedsol_loss_type": "kl",
    "fedsol_temperature": "3.0",
    "fedsol_adaptive": False,
    # 2. Optimization Policies & Learning Rates
    "client_lr_patience": "2",
    "client_lr_factor": "0.5",
    "client_lr_min": "1e-6",
    "momentum": "0.9",
    "weight_decay": "0.0001",
    "clip_grad_norm": "",
    # 3. Evaluation
    "local_test": True,
    "augment": False,
    # 4. Data Partitions, Hardware & System
    "partition_path": "",
    "client_dir": "",
    "pin_memory": True,
    "num_workers": "0",
}


class AdvancedConfigDialog(ctk.CTkToplevel):
    """Multi-card modal sub-window for granular configuration tuning."""

    def __init__(
        self,
        parent,
        mode: str = "server",  # "server" or "client"
        current_config: Optional[Dict[str, Any]] = None,
        active_strategy: str = "fedavg",
        on_save: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        super().__init__(parent)
        self.mode = mode.lower()
        self.active_strategy = active_strategy.lower()
        self.on_save = on_save

        # Copy defaults and overlay currently active config
        base_defaults = (
            DEFAULT_SERVER_ADVANCED if self.mode == "server" else DEFAULT_CLIENT_ADVANCED
        )
        self.defaults = copy.deepcopy(base_defaults)
        self.config_data = copy.deepcopy(base_defaults)
        if current_config:
            for k, v in current_config.items():
                if k in self.config_data:
                    self.config_data[k] = v

        # Widget mapping {field_name: (widget, type_str)}
        self.widgets: Dict[str, Any] = {}

        self._init_window(parent)
        self._build_ui()

    def _init_window(self, parent):
        title_tag = "FL Server" if self.mode == "server" else "FL Client"
        self.title(f"{title_tag} — Advanced Configuration & Hyperparameters")
        self.geometry("1060x780")
        self.minsize(920, 640)
        self.configure(fg_color=T.BACKGROUND)

        # Make modal & stay on top
        self.transient(parent)
        self.grab_set()

        # Center on parent window
        self.update_idletasks()
        try:
            px = parent.winfo_rootx()
            py = parent.winfo_rooty()
            pw = parent.winfo_width()
            ph = parent.winfo_height()
            x = px + max(0, (pw - 1060) // 2)
            y = py + max(0, (ph - 780) // 2)
            self.geometry(f"+{x}+{y}")
        except Exception:
            pass

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _build_ui(self):
        # 1. Header with clinical teal accent
        header = ctk.CTkFrame(self, height=72, corner_radius=0, fg_color=T.PANEL)
        header.pack(fill="x")
        header.pack_propagate(False)

        header_border = ctk.CTkFrame(self, height=1, corner_radius=0, fg_color=T.BORDER)
        header_border.pack(fill="x")

        title_container = ctk.CTkFrame(header, fg_color="transparent")
        title_container.pack(side="left", padx=20, pady=12)

        pill_accent = ctk.CTkFrame(
            title_container, width=4, height=40, corner_radius=2, fg_color=T.PRIMARY
        )
        pill_accent.pack(side="left", padx=(0, 12))

        text_box = ctk.CTkFrame(title_container, fg_color="transparent")
        text_box.pack(side="left")

        title_text = (
            "FL SERVER - ADVANCED CONFIGURATION"
            if self.mode == "server"
            else "FL CLIENT - ADVANCED CONFIGURATION"
        )
        sub_text = (
            "Fine-grained algorithmic hyperparameters, optimization policies, early stopping, and storage overrides"
            if self.mode == "server"
            else "Personalized client training parameters, hardware data loading, and partition overrides"
        )

        lbl_t = ctk.CTkLabel(
            text_box,
            text=title_text,
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=T.SIDEBAR,
        )
        lbl_t.pack(anchor="w")

        lbl_s = ctk.CTkLabel(
            text_box,
            text=sub_text,
            font=ctk.CTkFont(size=11),
            text_color=T.MUTED,
        )
        lbl_s.pack(anchor="w")

        btn_close_top = ctk.CTkButton(
            header,
            text="X",
            width=34,
            height=34,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=T.SURFACE,
            hover_color=T.BORDER,
            text_color=T.MUTED,
            command=self._on_cancel,
        )
        btn_close_top.pack(side="right", padx=20)

        # 2. SCROLLABLE BODY (2-Column Grid of 4 Cards)
        self.scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll.pack(fill="both", expand=True, padx=16, pady=12)

        cards_grid = ctk.CTkFrame(self.scroll, fg_color="transparent")
        cards_grid.pack(fill="both", expand=True)
        cards_grid.grid_columnconfigure((0, 1), weight=1, uniform="col")

        # Card 1: Strategy Hyperparameters
        self._build_card_strategy(cards_grid, row=0, col=0)

        # Card 2: Optimization Policies & LR
        self._build_card_optimization(cards_grid, row=0, col=1)

        # Card 3: Evaluation & Early Stopping
        self._build_card_evaluation(cards_grid, row=1, col=0)

        # Card 4: Hardware, Partitions & I/O
        self._build_card_system(cards_grid, row=1, col=1)

        # 3. BOTTOM ACTION BAR (White with hairline top border)
        footer_border = ctk.CTkFrame(self, height=1, corner_radius=0, fg_color=T.BORDER)
        footer_border.pack(fill="x")

        footer = ctk.CTkFrame(self, height=58, corner_radius=0, fg_color=T.PANEL)
        footer.pack(fill="x")
        footer.pack_propagate(False)

        btn_reset = ctk.CTkButton(
            footer,
            text="Restore Defaults",
            width=150,
            height=34,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=T.SURFACE,
            hover_color=T.SURFACE,
            text_color=T.MUTED,
            border_width=1,
            border_color=T.BORDER,
            command=self._on_restore_defaults,
        )
        btn_reset.pack(side="left", padx=20, pady=12)

        self.lbl_status = ctk.CTkLabel(
            footer,
            text="",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=T.ERROR,
        )
        self.lbl_status.pack(side="left", padx=10)

        btn_save = ctk.CTkButton(
            footer,
            text="Save & Apply",
            width=140,
            height=34,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=T.PRIMARY,
            hover_color=T.PRIMARY_HOVER,
            text_color=T.ON_ACCENT,
            command=self._on_save_clicked,
        )
        btn_save.pack(side="right", padx=20, pady=12)

        btn_cancel = ctk.CTkButton(
            footer,
            text="Cancel",
            width=95,
            height=34,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=T.PANEL,
            hover_color=T.SURFACE,
            text_color=T.MUTED,
            border_width=1,
            border_color=T.BORDER,
            command=self._on_cancel,
        )
        btn_cancel.pack(side="right", padx=(0, 6), pady=12)

    # --- CARD BUILDERS ---

    def _create_card(self, parent, title: str, icon: str, tag: str, row: int, col: int) -> ctk.CTkFrame:
        card = ctk.CTkFrame(
            parent,
            fg_color=T.PANEL,
            corner_radius=12,
            border_width=1,
            border_color=T.BORDER,
        )
        card.grid(row=row, column=col, sticky="nsew", padx=6, pady=6)

        c_header = ctk.CTkFrame(card, height=40, corner_radius=12, fg_color=T.SURFACE)
        c_header.pack(fill="x")
        c_header.pack_propagate(False)

        header_text = f"{icon}  {title}" if icon else title
        lbl = ctk.CTkLabel(
            c_header,
            text=header_text,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=T.TEXT,
        )
        lbl.pack(side="left", padx=12)

        tag_badge = ctk.CTkLabel(
            c_header,
            text=tag,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=T.ACCENT,
            fg_color=T.SELECTED,
            corner_radius=4,
            padx=9,
            pady=3,
        )
        tag_badge.pack(side="right", padx=12)

        sep = ctk.CTkFrame(card, height=1, fg_color=T.BORDER)
        sep.pack(fill="x")

        c_body = ctk.CTkFrame(card, fg_color="transparent")
        c_body.pack(fill="both", expand=True, padx=14, pady=12)
        return c_body

    def _add_entry_row(
        self,
        parent,
        field_key: str,
        label_text: str,
        default_val: str,
        is_highlight: bool = False,
    ):
        r = ctk.CTkFrame(parent, fg_color="transparent", height=34)
        r.pack(fill="x", pady=3)
        r.pack_propagate(False)

        lbl = ctk.CTkLabel(
            r,
            text=label_text,
            font=ctk.CTkFont(size=11, weight="bold" if is_highlight else "normal"),
            text_color=T.ACCENT if is_highlight else T.MUTED,
            width=180,
            anchor="w",
        )
        lbl.pack(side="left", padx=(2, 4))

        entry = ctk.CTkEntry(
            r,
            height=26,
            font=ctk.CTkFont(size=11),
            fg_color=T.PANEL,
            border_color=T.ACCENT if is_highlight else T.BORDER,
            text_color=T.TEXT,
        )
        entry.insert(0, str(self.config_data.get(field_key, default_val)))
        entry.pack(side="right", fill="x", expand=True, padx=(4, 2))
        self.widgets[field_key] = (entry, "str")
        return entry

    def _add_combo_row(
        self,
        parent,
        field_key: str,
        label_text: str,
        values: list[str],
        default_val: str,
    ):
        r = ctk.CTkFrame(parent, fg_color="transparent", height=34)
        r.pack(fill="x", pady=3)
        r.pack_propagate(False)

        lbl = ctk.CTkLabel(
            r,
            text=label_text,
            font=ctk.CTkFont(size=11),
            text_color=T.MUTED,
            width=180,
            anchor="w",
        )
        lbl.pack(side="left", padx=(2, 4))

        cmb = ctk.CTkComboBox(
            r,
            values=values,
            height=26,
            font=ctk.CTkFont(size=11),
            fg_color=T.PANEL,
            border_color=T.BORDER,
            text_color=T.TEXT,
        )
        curr = str(self.config_data.get(field_key, default_val))
        cmb.set(curr if curr in values else default_val)
        cmb.pack(side="right", fill="x", expand=True, padx=(4, 2))
        self.widgets[field_key] = (cmb, "combo")
        return cmb

    def _add_check_row(
        self,
        parent,
        field_key: str,
        label_text: str,
        default_checked: bool,
        subtext: Optional[str] = None,
    ):
        box = ctk.CTkFrame(parent, fg_color="transparent")
        box.pack(fill="x", pady=4)

        cb = ctk.CTkCheckBox(
            box,
            text=label_text,
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=T.TEXT,
            checkmark_color=T.PANEL,
            fg_color=T.PRIMARY,
            hover_color=T.PRIMARY_HOVER,
            border_color=T.MUTED,
        )
        if self.config_data.get(field_key, default_checked):
            cb.select()
        else:
            cb.deselect()
        cb.pack(anchor="w", padx=2, pady=(2, 0))

        if subtext:
            lbl_sub = ctk.CTkLabel(
                box,
                text=subtext,
                font=ctk.CTkFont(size=10),
                text_color=T.MUTED,
            )
            lbl_sub.pack(anchor="w", padx=(26, 0), pady=(0, 2))

        self.widgets[field_key] = (cb, "bool")
        return cb

    # --- CARD 1: STRATEGY HYPERPARAMETERS ---
    def _build_card_strategy(self, parent, row: int, col: int):
        body = self._create_card(
            parent,
            title="STRATEGY MATHEMATICAL HYPERPARAMETERS",
            icon="",
            tag="Algorithmic",
            row=row,
            col=col,
        )

        strat = self.active_strategy
        is_prox = "prox" in strat
        is_sol = "sol" in strat
        is_sw = "switch" in strat
        is_scr = "scr" in strat
        is_sld = "sld" in strat

        # Sub-header note
        note = ctk.CTkLabel(
            body,
            text=f"Active Strategy: {self.active_strategy.upper()}",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=T.ACCENT,
        )
        note.pack(anchor="w", padx=2, pady=(0, 6))

        # FedProx
        self._add_entry_row(
            body, "proximal_mu", "FedProx mu (μ):", "0.01", is_highlight=is_prox
        )

        # FedSOL
        self._add_entry_row(
            body, "fedsol_rho", "FedSOL rho (ρ):", "2.0", is_highlight=is_sol
        )
        self._add_combo_row(
            body, "fedsol_target", "FedSOL Target Layers:", ["head", "all"], "head"
        )
        self._add_combo_row(
            body, "fedsol_loss_type", "FedSOL Loss Type:", ["kl", "l2"], "kl"
        )
        self._add_entry_row(
            body, "fedsol_temperature", "FedSOL Temperature T:", "3.0"
        )
        self._add_check_row(
            body, "fedsol_adaptive", "FedSOL Adaptive Perturbation", False
        )

        if self.mode == "server":
            # FedSwitch
            self._add_entry_row(
                body,
                "switch_round",
                "FedSwitch Round (0=auto):",
                "0",
                is_highlight=is_sw,
            )
            self._add_combo_row(
                body,
                "fedswitch_weighting_mode",
                "FedSwitch Weighting Mode:",
                ["bounded_sum", "stratified"],
                "bounded_sum",
            )
            self._add_entry_row(
                body, "fedswitch_coverage", "FedSwitch Target Coverage ρ:", "0.9"
            )
            self._add_entry_row(
                body, "fedswitch_epsilon", "FedSwitch DP Epsilon ε:", "1.0"
            )
            self._add_entry_row(
                body, "fedswitch_c_bound", "FedSwitch Clipping Bound C:", "50"
            )

            # SCR
            self._add_entry_row(
                body, "scr_gamma", "SCR Class Support γ:", "1.0", is_highlight=is_scr
            )
            self._add_entry_row(
                body, "scr_head_lambda", "SCR Head Anchor λ:", "0.01"
            )

            # FedSLD
            self._add_entry_row(
                body, "fedsld_epsilon", "FedSLD Smoothing ε:", "1e-6", is_highlight=is_sld
            )
            self._add_combo_row(
                body, "fedsld_reduction", "FedSLD Loss Reduction:", ["mean", "sum"], "mean"
            )

    # --- CARD 2: OPTIMIZATION POLICIES & LR ---
    def _build_card_optimization(self, parent, row: int, col: int):
        body = self._create_card(
            parent,
            title="OPTIMIZATION POLICIES & LEARNING RATES",
            icon="",
            tag="Convergence",
            row=row,
            col=col,
        )

        if self.mode == "server":
            self._add_combo_row(
                body,
                "lr_mode",
                "LR Schedule Policy Mode:",
                ["fixed", "server_decay", "client_adaptive"],
                "fixed",
            )
            self._add_entry_row(
                body, "server_lr_decay_step", "Server LR Decay Step (Rounds):", "10"
            )
            self._add_entry_row(
                body, "server_lr_decay_gamma", "Server LR Decay Factor γ:", "0.5"
            )
            self._add_entry_row(body, "server_lr_min", "Server Minimum LR Floor:", "1e-5")

        self._add_entry_row(
            body, "client_lr_patience", "Client LR Plateau Patience:", "2"
        )
        self._add_entry_row(
            body, "client_lr_factor", "Client LR Reduction Factor:", "0.5"
        )
        self._add_entry_row(body, "client_lr_min", "Client Minimum LR Bound:", "1e-6")

        self._add_entry_row(body, "momentum", "SGD Momentum Parameter:", "0.9")
        self._add_entry_row(body, "weight_decay", "L2 Regularization Weight Decay:", "0.0001")
        self._add_entry_row(
            body, "clip_grad_norm", "Max Gradient L2 Norm Clip:", ""
        )

    # --- CARD 3: EVALUATION & EARLY STOPPING ---
    def _build_card_evaluation(self, parent, row: int, col: int):
        body = self._create_card(
            parent,
            title="EVALUATION & EARLY STOPPING",
            icon="",
            tag="Validation",
            row=row,
            col=col,
        )

        if self.mode == "server":
            self._add_check_row(
                body,
                "server_eval",
                "Enable Centralized Server Evaluation",
                False,
                "Evaluates aggregated global weights on central test split each round (Auto-disabled for FedBN)",
            )
            self._add_check_row(
                body,
                "local_test",
                "Enable Client Local Test Evaluation (BMCTA / BTA)",
                True,
                "Tracks personalized accuracy across clients' local test partitions (Recommended for FedBN)",
            )
            self._add_entry_row(
                body, "early_stop_patience", "Early Stop Patience (0=off):", "0"
            )
            self._add_combo_row(
                body,
                "early_stop_metric",
                "Early Stop Target Metric:",
                ["accuracy", "loss"],
                "accuracy",
            )
        else:
            self._add_check_row(
                body,
                "local_test",
                "Enable Client Local Test Evaluation (BMCTA / BTA)",
                True,
                "Evaluate model locally on client test partition",
            )
            self._add_check_row(
                body,
                "augment",
                "Data Augmentation (Random Crop / Horizontal Flip)",
                False,
                "Expands input diversity during local SGD epochs",
            )

    # --- CARD 4: DATA PARTITIONS, HARDWARE & SYSTEM ---
    def _build_card_system(self, parent, row: int, col: int):
        body = self._create_card(
            parent,
            title="DATA PARTITIONS, HARDWARE & SYSTEM",
            icon="",
            tag="System / IO",
            row=row,
            col=col,
        )

        # Partition Path Override
        r_part = ctk.CTkFrame(body, fg_color="transparent", height=34)
        r_part.pack(fill="x", pady=3)
        r_part.pack_propagate(False)

        lbl_p = ctk.CTkLabel(
            r_part,
            text="Partition Path Override:",
            font=ctk.CTkFont(size=11),
            text_color=T.MUTED,
            width=180,
            anchor="w",
        )
        lbl_p.pack(side="left", padx=(2, 4))

        e_part = ctk.CTkEntry(
            r_part,
            height=26,
            font=ctk.CTkFont(family="monospace", size=10),
            fg_color=T.PANEL,
            border_color=T.BORDER,
            text_color=T.TEXT,
            placeholder_text="data/partitions/... (Optional)",
        )
        curr_part = str(self.config_data.get("partition_path", ""))
        if curr_part:
            e_part.insert(0, curr_part)
        e_part.pack(side="left", fill="x", expand=True, padx=(4, 4))
        self.widgets["partition_path"] = (e_part, "str")

        btn_browse_part = ctk.CTkButton(
            r_part,
            text="Browse",
            width=70,
            height=26,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=T.PANEL,
            hover_color=T.SURFACE,
            text_color=T.MUTED,
            border_width=1,
            border_color=T.BORDER,
            command=lambda: self._browse_file(e_part, "Select Partition JSON", "*.json"),
        )
        btn_browse_part.pack(side="right", padx=(2, 2))

        # Output / Save Directory
        r_dir = ctk.CTkFrame(body, fg_color="transparent", height=34)
        r_dir.pack(fill="x", pady=3)
        r_dir.pack_propagate(False)

        dir_key = "save_dir" if self.mode == "server" else "client_dir"
        dir_label = "Custom Results Save Dir:" if self.mode == "server" else "Custom Client Dir:"

        lbl_d = ctk.CTkLabel(
            r_dir,
            text=dir_label,
            font=ctk.CTkFont(size=11),
            text_color=T.MUTED,
            width=180,
            anchor="w",
        )
        lbl_d.pack(side="left", padx=(2, 4))

        e_dir = ctk.CTkEntry(
            r_dir,
            height=26,
            font=ctk.CTkFont(family="monospace", size=10),
            fg_color=T.PANEL,
            border_color=T.BORDER,
            text_color=T.TEXT,
            placeholder_text="results/federated/... (Optional)",
        )
        curr_dir = str(self.config_data.get(dir_key, ""))
        if curr_dir:
            e_dir.insert(0, curr_dir)
        e_dir.pack(side="left", fill="x", expand=True, padx=(4, 4))
        self.widgets[dir_key] = (e_dir, "str")

        btn_browse_dir = ctk.CTkButton(
            r_dir,
            text="Browse",
            width=70,
            height=26,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=T.PANEL,
            hover_color=T.SURFACE,
            text_color=T.MUTED,
            border_width=1,
            border_color=T.BORDER,
            command=lambda: self._browse_dir(e_dir, "Select Output Directory"),
        )
        btn_browse_dir.pack(side="right", padx=(2, 2))

        # Pinned host memory
        self._add_check_row(
            body,
            "pin_memory",
            "DataLoader Pin Host Memory (CUDA Fast Transfer)",
            True,
            "Locks page memory for faster GPU tensor copy",
        )

        # Worker count
        self._add_entry_row(
            body,
            "num_workers",
            "DataLoader Worker Processes:",
            "0",
        )

    # --- BROWSER DIALOGS ---

    def _browse_file(self, target_entry, title: str, pattern: str):
        from tkinter import filedialog

        init_dir = str(PROJECT_ROOT / "data" / "partitions")
        f = filedialog.askopenfilename(
            parent=self,
            initialdir=init_dir,
            title=title,
            filetypes=[("Target Files", pattern), ("All Files", "*.*")],
        )
        if f:
            target_entry.delete(0, "end")
            target_entry.insert(0, f)

    def _browse_dir(self, target_entry, title: str):
        from tkinter import filedialog

        init_dir = str(PROJECT_ROOT / "results")
        d = filedialog.askdirectory(parent=self, initialdir=init_dir, title=title)
        if d:
            target_entry.delete(0, "end")
            target_entry.insert(0, d)

    # --- ACTION HANDLERS ---

    def _on_restore_defaults(self):
        """Restores all input fields to their base defaults."""
        for key, (widget, w_type) in self.widgets.items():
            if key not in self.defaults:
                continue
            default_v = self.defaults[key]
            if w_type == "str":
                widget.delete(0, "end")
                widget.insert(0, str(default_v))
            elif w_type == "combo":
                widget.set(str(default_v))
            elif w_type == "bool":
                if default_v:
                    widget.select()
                else:
                    widget.deselect()
        self.lbl_status.configure(text="Restored all parameters to standard defaults.", text_color=T.ACCENT)

    def _on_cancel(self):
        self.grab_release()
        self.destroy()

    def _on_save_clicked(self):
        """Validates all inputs, commits to config dict, and fires callback."""
        cleaned_config: Dict[str, Any] = {}
        errors = []

        for key, (widget, w_type) in self.widgets.items():
            if w_type == "str":
                val = widget.get().strip()
                # Numeric checks for specific keys
                if key in (
                    "proximal_mu",
                    "fedsol_rho",
                    "fedsol_temperature",
                    "fedswitch_coverage",
                    "fedswitch_epsilon",
                    "scr_gamma",
                    "scr_head_lambda",
                    "fedsld_epsilon",
                    "server_lr_decay_gamma",
                    "server_lr_min",
                    "client_lr_factor",
                    "client_lr_min",
                    "momentum",
                    "weight_decay",
                ):
                    if val != "":
                        try:
                            float(val)
                        except ValueError:
                            errors.append(f"{key} must be a valid float")
                elif key in (
                    "switch_round",
                    "fedswitch_c_bound",
                    "server_lr_decay_step",
                    "client_lr_patience",
                    "early_stop_patience",
                    "num_workers",
                ):
                    if val != "":
                        try:
                            int(val)
                        except ValueError:
                            errors.append(f"{key} must be a valid integer")
                cleaned_config[key] = val
            elif w_type == "combo":
                cleaned_config[key] = widget.get()
            elif w_type == "bool":
                cleaned_config[key] = bool(widget.get())

        if errors:
            self.lbl_status.configure(
                text=f"Validation Error: {errors[0]}", text_color=T.ERROR
            )
            return

        # Deliver to callback
        if self.on_save:
            self.on_save(cleaned_config)

        self.grab_release()
        self.destroy()
