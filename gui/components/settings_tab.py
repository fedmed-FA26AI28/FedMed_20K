import copy
import tkinter as tk
import customtkinter as ctk
from typing import Dict, Any, Optional, Callable
from gui import theme as T
from gui.components.advanced_config_dialog import (
    DEFAULT_SERVER_ADVANCED,
    DEFAULT_CLIENT_ADVANCED,
)


class SettingsTab(ctk.CTkFrame):
    """Embedded 4-card Settings & Hyperparameters Tab for the dashboard."""

    def __init__(
        self,
        master,
        current_config: Optional[Dict[str, Any]] = None,
        on_save: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs,
    ):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.config: Dict[str, Any] = copy.deepcopy(DEFAULT_SERVER_ADVANCED)
        if current_config:
            self.config.update(current_config)
        self.on_save = on_save
        self.fields: Dict[str, Any] = {}

        self._build_ui()

    def _build_ui(self):
        # 1. Header Banner
        header = ctk.CTkFrame(self, fg_color=T.PANEL, corner_radius=8, border_width=1, border_color=T.BORDER, height=52)
        header.pack(fill="x", padx=4, pady=(4, 6))
        header.pack_propagate(False)

        lbl_title = ctk.CTkLabel(
            header,
            text="SETTINGS & HYPERPARAMETER CONFIGURATION",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=T.TEXT,
        )
        lbl_title.pack(side="left", padx=16, pady=10)

        lbl_sub = ctk.CTkLabel(
            header,
            text="Deep algorithmic hyperparameters, optimization policies, evaluation, and system paths",
            font=ctk.CTkFont(size=12),
            text_color=T.MUTED,
        )
        lbl_sub.pack(side="left", padx=8, pady=10)

        # 2. 4-Card Scrollable Grid
        self.scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll.pack(fill="both", expand=True, padx=4, pady=2)

        cards_grid = ctk.CTkFrame(self.scroll, fg_color="transparent")
        cards_grid.pack(fill="both", expand=True)
        cards_grid.grid_columnconfigure((0, 1), weight=1, uniform="col")

        # Card 1: Strategy Hyperparameters
        self._build_card(
            cards_grid,
            title="STRATEGY HYPERPARAMETERS",
            row=0,
            col=0,
            fields_def=[
                ("proximal_mu", "FedProx Proximal Mu (u):", ctk.CTkEntry, "0.01"),
                ("fedsol_rho", "FedSOL Perturbation Rho (p):", ctk.CTkEntry, "2.0"),
                ("fedsol_target", "FedSOL Perturb Target:", ctk.CTkComboBox, "head", ["head", "backbone", "all"]),
                ("fedsol_loss_type", "FedSOL Distillation Loss:", ctk.CTkComboBox, "kl", ["kl", "trades"]),
                ("fedsol_temperature", "FedSOL Temperature (T):", ctk.CTkEntry, "3.0"),
                ("fedsol_adaptive", "FedSOL Adaptive Perturbation:", ctk.CTkCheckBox, False),
                ("switch_round", "FedSwitch Transition Round:", ctk.CTkEntry, "0"),
                ("fedswitch_coverage", "FedSwitch Target Coverage:", ctk.CTkEntry, "0.9"),
                ("scr_gamma", "SCR Class Weight Gamma:", ctk.CTkEntry, "1.0"),
                ("fedsld_epsilon", "FedSLD Smoothing Epsilon:", ctk.CTkEntry, "1e-6"),
            ],
        )

        # Card 2: Optimization Policies & LR
        self._build_card(
            cards_grid,
            title="OPTIMIZER & LR SCHEDULING",
            row=0,
            col=1,
            fields_def=[
                ("server_lr_decay_step", "Server Decay Step (Rounds):", ctk.CTkEntry, "10"),
                ("server_lr_decay_gamma", "Server Decay Gamma:", ctk.CTkEntry, "0.5"),
                ("client_lr_patience", "Client Reducer Patience:", ctk.CTkEntry, "2"),
                ("client_lr_factor", "Client Reducer Factor:", ctk.CTkEntry, "0.5"),
                ("momentum", "SGD Momentum:", ctk.CTkEntry, "0.9"),
                ("weight_decay", "L2 Regularization Weight Decay:", ctk.CTkEntry, "0.0001"),
                ("clip_grad_norm", "Gradient Clipping Norm:", ctk.CTkEntry, ""),
            ],
        )

        # Card 3: Evaluation & Early Stopping
        self._build_card(
            cards_grid,
            title="EVALUATION & EARLY STOPPING",
            row=1,
            col=0,
            fields_def=[
                ("server_eval", "Server Centralized Evaluation:", ctk.CTkCheckBox, False),
                ("local_test", "Client Local Test Evaluation:", ctk.CTkCheckBox, True),
                ("early_stop_patience", "Early Stopping Patience:", ctk.CTkEntry, "0"),
                ("early_stop_metric", "Early Stopping Target Metric:", ctk.CTkComboBox, "accuracy", ["accuracy", "loss"]),
            ],
        )

        # Card 4: Hardware & Storage Paths
        self._build_card(
            cards_grid,
            title="STORAGE & SYSTEM PATHS",
            row=1,
            col=1,
            fields_def=[
                ("partition_path", "Partition JSON Path:", ctk.CTkEntry, ""),
                ("save_dir", "Custom Results Save Directory:", ctk.CTkEntry, ""),
                ("pin_memory", "CUDA Pin Memory:", ctk.CTkCheckBox, True),
                ("num_workers", "PyTorch DataLoader Workers:", ctk.CTkEntry, "0"),
            ],
        )

        # 3. Footer Action Bar
        footer = ctk.CTkFrame(self, fg_color=T.PANEL, corner_radius=8, border_width=1, border_color=T.BORDER, height=48)
        footer.pack(fill="x", padx=4, pady=(6, 4))
        footer.pack_propagate(False)

        btn_restore = ctk.CTkButton(
            footer,
            text="Restore Defaults",
            height=34,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=T.SURFACE,
            hover_color=T.BORDER,
            text_color=T.MUTED,
            border_width=1,
            border_color=T.BORDER,
            command=self._on_restore_defaults,
        )
        btn_restore.pack(side="left", padx=14, pady=8)

        self.lbl_status = ctk.CTkLabel(
            footer,
            text="",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=T.SUCCESS,
        )
        self.lbl_status.pack(side="left", padx=10)

        btn_apply = ctk.CTkButton(
            footer,
            text="Save & Apply Settings",
            height=34,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=T.PRIMARY,
            hover_color=T.PRIMARY_HOVER,
            text_color=T.ON_ACCENT,
            command=self._on_apply,
        )
        btn_apply.pack(side="right", padx=14, pady=8)

    def _build_card(self, parent, title: str, row: int, col: int, fields_def: list):
        card = ctk.CTkFrame(parent, fg_color=T.PANEL, corner_radius=8, border_width=1, border_color=T.BORDER)
        card.grid(row=row, column=col, sticky="nsew", padx=6, pady=6)

        lbl_head = ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=T.TEXT,
        )
        lbl_head.pack(anchor="w", padx=14, pady=(12, 6))

        div = ctk.CTkFrame(card, height=1, fg_color=T.BORDER)
        div.pack(fill="x", padx=14, pady=(0, 8))

        for item in fields_def:
            key, label_txt, w_cls, def_val = item[0], item[1], item[2], item[3]
            extra_vals = item[4] if len(item) > 4 else None

            row_frame = ctk.CTkFrame(card, fg_color="transparent", height=34)
            row_frame.pack(fill="x", padx=14, pady=3)
            row_frame.pack_propagate(False)

            lbl = ctk.CTkLabel(
                row_frame,
                text=label_txt,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=T.TEXT,
                width=220,
                anchor="w",
            )
            lbl.pack(side="left")

            val = self.config.get(key, def_val)

            if w_cls == ctk.CTkCheckBox:
                var = tk.BooleanVar(value=bool(val))
                w = ctk.CTkCheckBox(
                    row_frame,
                    text="",
                    variable=var,
                    width=22,
                    height=22,
                    checkbox_width=20,
                    checkbox_height=20,
                    fg_color=T.PRIMARY,
                    hover_color=T.PRIMARY_HOVER,
                )
                w.pack(side="right", padx=4)
                self.fields[key] = var
            elif w_cls == ctk.CTkComboBox:
                w = ctk.CTkComboBox(
                    row_frame,
                    values=extra_vals or [],
                    height=28,
                    width=160,
                    font=ctk.CTkFont(size=12),
                    fg_color=T.SURFACE,
                    border_color=T.BORDER,
                    text_color=T.TEXT,
                )
                w.set(str(val))
                w.pack(side="right")
                self.fields[key] = w
            else:
                w = ctk.CTkEntry(
                    row_frame,
                    height=28,
                    width=160,
                    font=ctk.CTkFont(family=T.MONO_FONT, size=12),
                    fg_color=T.SURFACE,
                    border_color=T.BORDER,
                    text_color=T.TEXT,
                )
                w.insert(0, str(val))
                w.pack(side="right")
                self.fields[key] = w

    def _on_apply(self):
        new_cfg = {}
        for k, w in self.fields.items():
            if isinstance(w, tk.BooleanVar):
                new_cfg[k] = w.get()
            elif isinstance(w, (ctk.CTkEntry, ctk.CTkComboBox)):
                new_cfg[k] = w.get().strip()
        self.config.update(new_cfg)
        if self.on_save:
            self.on_save(new_cfg)
        self.lbl_status.configure(text="[OK] Settings Successfully Saved & Applied")
        self.after(3000, lambda: self.lbl_status.configure(text=""))

    def _on_restore_defaults(self):
        for k, v in DEFAULT_SERVER_ADVANCED.items():
            if k in self.fields:
                w = self.fields[k]
                if isinstance(w, tk.BooleanVar):
                    w.set(bool(v))
                elif isinstance(w, ctk.CTkEntry):
                    w.delete(0, tk.END)
                    w.insert(0, str(v))
                elif isinstance(w, ctk.CTkComboBox):
                    w.set(str(v))
        self.lbl_status.configure(text="[OK] Restored Standard Defaults", text_color=T.MUTED)
        self.after(3000, lambda: self.lbl_status.configure(text=""))
