import copy
import tkinter as tk
import customtkinter as ctk
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List
from gui.config_manager import (
    PROJECT_ROOT,
    DATASETS,
    STRATEGIES,
    MODELS,
    LR_MODES,
    OPTIMIZERS,
)
from gui.components.advanced_config_dialog import (
    DEFAULT_SERVER_ADVANCED,
    AdvancedConfigDialog,
)


class ServerTab(ctk.CTkFrame):
    """FL Server Control tab: parameter forms, advanced sub-window trigger, and action buttons."""

    def __init__(
        self,
        master,
        on_start: Optional[Callable[[List[str]], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
        on_export: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_export = on_export

        self.advanced_config: Dict[str, Any] = copy.deepcopy(DEFAULT_SERVER_ADVANCED)
        self.adv_expanded = False
        self._build_form()

    def _build_form(self):
        """Builds scrollable configuration form on the left with rock-solid row alignment."""
        self.scroll = ctk.CTkScrollableFrame(self, fg_color="#ffffff", corner_radius=8, border_width=1, border_color="#e8edf2", width=380)
        self.scroll.pack(fill="both", expand=True, padx=4, pady=4)

        title = ctk.CTkLabel(
            self.scroll,
            text="⚙️ SERVER PARAMETER CONFIGURATION",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#162d43",
        )
        title.pack(anchor="w", padx=12, pady=(10, 14))

        self.fields: Dict[str, Any] = {}

        # Safe row builder: ALWAYS parents both label and widget to the row frame
        def add_field(label_text: str, widget_cls, **kwargs):
            row = ctk.CTkFrame(self.scroll, fg_color="transparent", height=32)
            row.pack(fill="x", padx=10, pady=4)
            row.pack_propagate(False)

            lbl = ctk.CTkLabel(
                row,
                text=label_text,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color="#475569",
                width=140,
                anchor="w",
            )
            lbl.pack(side="left", padx=(4, 8))

            # Crucial: widget master is row, NOT self.scroll!
            kwargs.setdefault("fg_color", "#ffffff")
            kwargs.setdefault("border_color", "#dce3eb")
            kwargs.setdefault("text_color", "#162d43")
            widget = widget_cls(row, **kwargs)
            widget.pack(side="right", fill="x", expand=True, padx=(0, 4))
            return widget

        # 1. Dataset
        self.fields["dataset"] = add_field(
            "Dataset:",
            ctk.CTkComboBox,
            values=DATASETS,
            height=28,
            font=ctk.CTkFont(size=11),
        )
        self.fields["dataset"].set("pathmnist")

        # 2. Strategy
        self.fields["strategy"] = add_field(
            "Strategy:",
            ctk.CTkComboBox,
            values=STRATEGIES,
            height=28,
            font=ctk.CTkFont(size=11),
            command=self._on_strategy_changed,
        )
        self.fields["strategy"].set("fedbns")

        # 3. Model Architecture
        self.fields["model"] = add_field(
            "Model Architecture:",
            ctk.CTkComboBox,
            values=MODELS,
            height=28,
            font=ctk.CTkFont(size=11),
        )
        self.fields["model"].set("baseline_bn")

        # 4. Optimizer
        self.fields["optimizer"] = add_field(
            "Optimizer:",
            ctk.CTkComboBox,
            values=OPTIMIZERS,
            height=28,
            font=ctk.CTkFont(size=11),
        )
        self.fields["optimizer"].set("sgd")

        # 5. Rounds
        self.fields["rounds"] = add_field("FL Rounds:", ctk.CTkEntry, height=28, font=ctk.CTkFont(size=11))
        self.fields["rounds"].insert(0, "40")

        # 6. Min Clients
        self.fields["min_clients"] = add_field("Min Clients:", ctk.CTkEntry, height=28, font=ctk.CTkFont(size=11))
        self.fields["min_clients"].insert(0, "12")

        # 7. Local Epochs
        self.fields["local_epochs"] = add_field("Local Epochs:", ctk.CTkEntry, height=28, font=ctk.CTkFont(size=11))
        self.fields["local_epochs"].insert(0, "5")

        # 8. Base LR
        self.fields["learning_rate"] = add_field("Learning Rate:", ctk.CTkEntry, height=28, font=ctk.CTkFont(size=11))
        self.fields["learning_rate"].insert(0, "0.01")

        # 9. LR Mode
        self.fields["lr_mode"] = add_field(
            "LR Policy Mode:",
            ctk.CTkComboBox,
            values=LR_MODES,
            height=28,
            font=ctk.CTkFont(size=11),
        )
        self.fields["lr_mode"].set("fixed")

        # 10. Seed
        self.fields["seed"] = add_field("Random Seed:", ctk.CTkEntry, height=28, font=ctk.CTkFont(size=11))
        self.fields["seed"].insert(0, "42")

        # 11. Run Name
        self.fields["name"] = add_field("Run Name / Tag:", ctk.CTkEntry, height=28, font=ctk.CTkFont(size=11))
        self.fields["name"].insert(0, "gui_server_run")

        # --- ADVANCED CONFIGURATION SUB-WINDOW TRIGGER ---
        self.btn_open_adv = ctk.CTkButton(
            self.scroll,
            text="⚙  Open Advanced Configuration...",
            height=34,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#f1f5f9",
            hover_color="#e2e8f0",
            text_color="#162d43",
            border_width=1,
            border_color="#dfe6ed",
            command=self._open_advanced_dialog,
        )
        self.btn_open_adv.pack(fill="x", padx=10, pady=(14, 4))

        self.lbl_adv_status = ctk.CTkLabel(
            self.scroll,
            text="● Standard Defaults Active",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#64748b",
            anchor="center",
        )
        self.lbl_adv_status.pack(fill="x", padx=10, pady=(0, 6))

        # Backward compatibility placeholders
        self.adv_container = ctk.CTkFrame(self.scroll, fg_color="#f8fafc", corner_radius=6)
        self.btn_toggle_adv = self.btn_open_adv

        # --- ACTION BUTTONS ---
        btn_box = ctk.CTkFrame(self.scroll, fg_color="transparent")
        btn_box.pack(fill="x", padx=10, pady=(18, 12))

        self.btn_start = ctk.CTkButton(
            btn_box,
            text="▶ START FL SERVER",
            height=44,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#0c8c80",
            hover_color="#087b70",
            text_color="#ffffff",
            command=self._on_start_click,
        )
        self.btn_start.pack(fill="x", pady=(0, 8))

        sub_btns = ctk.CTkFrame(btn_box, fg_color="transparent")
        sub_btns.pack(fill="x")

        self.btn_stop = ctk.CTkButton(
            sub_btns,
            text="■ STOP SERVER",
            height=34,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#ba4c4c",
            hover_color="#991b1b",
            text_color="#ffffff",
            command=self._on_stop_click,
        )
        self.btn_stop.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.btn_export = ctk.CTkButton(
            sub_btns,
            text="📊 Export Results",
            height=34,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#ffffff",
            hover_color="#f3f7fa",
            text_color="#42566b",
            border_width=1,
            border_color="#dfe6ed",
            command=self._on_export_click,
        )
        self.btn_export.pack(side="right", fill="x", expand=True, padx=(4, 0))

    def _open_advanced_dialog(self):
        """Opens the modular multi-card Advanced Configuration sub-window."""
        strat = self.fields["strategy"].get()
        AdvancedConfigDialog(
            parent=self.winfo_toplevel(),
            mode="server",
            current_config=self.advanced_config,
            active_strategy=strat,
            on_save=self._on_advanced_saved,
        )

    def _on_advanced_saved(self, new_cfg: Dict[str, Any]):
        """Persists updated advanced configuration and updates visual indicator."""
        self.advanced_config.update(new_cfg)
        overrides = 0
        for k, v in self.advanced_config.items():
            if k in DEFAULT_SERVER_ADVANCED and str(v) != str(DEFAULT_SERVER_ADVANCED[k]):
                overrides += 1
        if overrides > 0:
            self.lbl_adv_status.configure(
                text=f"● {overrides} Custom Overrides Active",
                text_color="#0c8c80",
            )
            self.btn_open_adv.configure(border_color="#0c8c80")
        else:
            self.lbl_adv_status.configure(
                text="● Standard Defaults Active",
                text_color="#64748b",
            )
            self.btn_open_adv.configure(border_color="#dfe6ed")

    def _toggle_advanced(self):
        """Backward compatibility helper for headless test suites."""
        self.adv_expanded = not self.adv_expanded
        if self.adv_expanded:
            self.btn_toggle_adv.configure(text="⚙  Advanced Configuration (Active)")
        else:
            self.btn_toggle_adv.configure(text="⚙  Open Advanced Configuration...")

    def _on_strategy_changed(self, choice: str):
        pass

    def build_command(self, python_exe: str, port: str = "8080") -> List[str]:
        """Compile parameters into exact server.server CLI invocation command."""
        cmd = [
            python_exe,
            "-m",
            "server.server",
            "--dataset", self.fields["dataset"].get(),
            "--strategy", self.fields["strategy"].get(),
            "--model", self.fields["model"].get(),
            "--optimizer", self.fields["optimizer"].get(),
            "--rounds", self.fields["rounds"].get().strip(),
            "--min_clients", self.fields["min_clients"].get().strip(),
            "--local_epochs", self.fields["local_epochs"].get().strip(),
            "--learning_rate", self.fields["learning_rate"].get().strip(),
            "--lr_mode", self.fields["lr_mode"].get(),
            "--seed", self.fields["seed"].get().strip(),
            "--name", self.fields["name"].get().strip(),
            "--port", str(port),
            "--host", "0.0.0.0",
        ]

        adv = self.advanced_config

        # 1. Partition Path Override
        part = adv.get("partition_path", "").strip()
        if part:
            cmd.extend(["--partition_path", part])

        # 2. Custom Results Save Directory
        s_dir = adv.get("save_dir", "").strip()
        if s_dir:
            cmd.extend(["--save_dir", s_dir])

        # 3. Strategy Mathematical Hyperparameters
        strat = self.fields["strategy"].get().lower()
        if "prox" in strat:
            p_mu = str(adv.get("proximal_mu", "0.01")).strip()
            if p_mu:
                cmd.extend(["--proximal_mu", p_mu])
        elif "sol" in strat:
            cmd.extend(["--fedsol_rho", str(adv.get("fedsol_rho", "2.0"))])
            cmd.extend(["--fedsol_target", str(adv.get("fedsol_target", "head"))])
            cmd.extend(["--fedsol_loss_type", str(adv.get("fedsol_loss_type", "kl"))])
            cmd.extend(["--fedsol_temperature", str(adv.get("fedsol_temperature", "3.0"))])
            if adv.get("fedsol_adaptive", False):
                cmd.append("--fedsol_adaptive")
        elif "switch" in strat:
            sw_r = str(adv.get("switch_round", "0")).strip()
            if sw_r and int(sw_r) > 0:
                cmd.extend(["--switch_round", sw_r])
            cmd.extend(["--fedswitch_weighting_mode", str(adv.get("fedswitch_weighting_mode", "bounded_sum"))])
            cmd.extend(["--fedswitch_coverage", str(adv.get("fedswitch_coverage", "0.9"))])
            cmd.extend(["--fedswitch_epsilon", str(adv.get("fedswitch_epsilon", "1.0"))])
            cmd.extend(["--fedswitch_c_bound", str(adv.get("fedswitch_c_bound", "50"))])
        elif "scr" in strat:
            cmd.extend(["--scr_gamma", str(adv.get("scr_gamma", "1.0"))])
            cmd.extend(["--scr_head_lambda", str(adv.get("scr_head_lambda", "0.01"))])
        elif "sld" in strat:
            cmd.extend(["--fedsld_epsilon", str(adv.get("fedsld_epsilon", "1e-6"))])
            cmd.extend(["--fedsld_reduction", str(adv.get("fedsld_reduction", "mean"))])

        # 4. Optimizer & LR Policies
        opt = self.fields["optimizer"].get()
        if opt == "sgd":
            mom = str(adv.get("momentum", "0.9")).strip()
            if mom:
                cmd.extend(["--momentum", mom])
        wd = str(adv.get("weight_decay", "")).strip()
        if wd:
            try:
                if float(wd) > 0:
                    cmd.extend(["--weight_decay", wd])
            except ValueError:
                pass
        cgn = str(adv.get("clip_grad_norm", "")).strip()
        if cgn:
            cmd.extend(["--clip_grad_norm", cgn])

        # 5. Evaluation & Early Stopping
        if not adv.get("server_eval", True):
            cmd.append("--no-server_eval")
        if adv.get("local_test", False):
            cmd.append("--local_test")

        es_p = str(adv.get("early_stop_patience", "0")).strip()
        if es_p and int(es_p) > 0:
            cmd.extend(["--early_stop_patience", es_p])
            cmd.extend(["--early_stop_metric", str(adv.get("early_stop_metric", "accuracy"))])

        return cmd

    def _on_start_click(self):
        if self.on_start:
            self.on_start(self.build_command(python_exe=""))

    def _on_stop_click(self):
        if self.on_stop:
            self.on_stop()

    def _on_export_click(self):
        if self.on_export:
            self.on_export()

    def set_running_state(self, running: bool):
        if running:
            self.btn_start.configure(state="disabled", fg_color="#374151")
            self.btn_stop.configure(state="normal", fg_color="#dc2626")
        else:
            self.btn_start.configure(state="normal", fg_color="#059669")
            self.btn_stop.configure(state="disabled", fg_color="#374151")
