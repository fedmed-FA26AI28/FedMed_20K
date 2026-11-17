from gui import theme as T
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
    PARTITION_MODES,
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
        on_mode_change: Optional[Callable[[str], None]] = None,
        on_advanced_saved: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs,
    ):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_export = on_export
        self.on_mode_change = on_mode_change
        self.on_advanced_saved_cb = on_advanced_saved

        self.run_mode: str = "standalone"  # "standalone" or "simulation"
        self._is_running: bool = False
        self.advanced_config: Dict[str, Any] = copy.deepcopy(DEFAULT_SERVER_ADVANCED)
        self.adv_expanded = False
        self._build_form()

    def _build_form(self):
        """Builds scrollable configuration form on the left with rock-solid row alignment."""
        self.scroll = ctk.CTkScrollableFrame(self, fg_color=T.PANEL, corner_radius=12, border_width=1, border_color=T.BORDER, width=300)
        self.scroll.pack(fill="both", expand=True, padx=4, pady=4)

        title = ctk.CTkLabel(
            self.scroll,
            text="Federation configuration",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=T.TEXT,
        )
        title.pack(anchor="w", padx=12, pady=(10, 8))

        self.seg_run_mode = ctk.CTkSegmentedButton(
            self.scroll,
            values=["Standalone", "Simulation"],
            height=30,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=T.SURFACE,
            selected_color=T.PRIMARY,
            selected_hover_color=T.PRIMARY_HOVER,
            unselected_color=T.SURFACE,
            unselected_hover_color=T.BORDER,
            text_color=T.TEXT,
            command=self._on_run_mode_changed,
        )
        self.seg_run_mode.set("Standalone")
        self.seg_run_mode.pack(fill="x", padx=10, pady=(0, 8))

        self.fields: Dict[str, Any] = {}
        self.field_rows: Dict[str, ctk.CTkFrame] = {}

        # Safe row builder: ALWAYS parents both label and widget to the row frame
        def add_field(label_text: str, widget_cls, **kwargs):
            row = ctk.CTkFrame(self.scroll, fg_color="transparent", height=32)
            row.pack(fill="x", padx=10, pady=4)
            row.pack_propagate(False)

            lbl = ctk.CTkLabel(
                row,
                text=label_text,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=T.MUTED,
                width=120,
                anchor="w",
            )
            lbl.pack(side="left", padx=(4, 8))

            # Crucial: widget master is row, NOT self.scroll!
            kwargs.setdefault("fg_color", T.PANEL)
            kwargs.setdefault("border_color", T.BORDER)
            kwargs.setdefault("text_color", T.TEXT)
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
        self._min_clients_row = self.fields["min_clients"].master
        self.field_rows["min_clients"] = self._min_clients_row

        # 6B. Partition Mode (important in both Standalone and Simulation modes for SCR/FedSLD/FedSwitch & recorder)
        self.fields["partition_mode"] = add_field(
            "Partition Mode:",
            ctk.CTkComboBox,
            values=PARTITION_MODES,
            height=28,
            font=ctk.CTkFont(size=11),
        )
        self.fields["partition_mode"].set("practical")
        self.field_rows["partition_mode"] = self.fields["partition_mode"].master

        # 6C. Simulation-only fields (hidden in Standalone Server mode)
        self.fields["batch_size"] = add_field("Batch Size:", ctk.CTkEntry, height=28, font=ctk.CTkFont(size=11))
        self.fields["batch_size"].insert(0, "256")
        self.field_rows["batch_size"] = self.fields["batch_size"].master

        self.fields["client_execution"] = add_field(
            "Client Execution:",
            ctk.CTkComboBox,
            values=["Sequential GPU Lock", "Parallel Clients"],
            height=28,
            font=ctk.CTkFont(size=11),
        )
        self.fields["client_execution"].set("Sequential GPU Lock")
        self.field_rows["client_execution"] = self.fields["client_execution"].master

        self._sim_row_frames = [
            self.field_rows["batch_size"],
            self.field_rows["client_execution"],
        ]
        for row_frame in self._sim_row_frames:
            row_frame.pack_forget()

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

        # --- ADVANCED CONFIGURATION (managed via global Settings Tab; keep attributes unmounted for compatibility) ---
        self.btn_open_adv = ctk.CTkButton(
            self.scroll,
            text="Open Advanced Configuration...",
            height=34,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=T.SURFACE,
            hover_color=T.BORDER,
            text_color=T.TEXT,
            border_width=1,
            border_color=T.BORDER,
            command=self._open_advanced_dialog,
        )

        self.lbl_adv_status = ctk.CTkLabel(
            self.scroll,
            text="[Standard Defaults Active]",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=T.MUTED,
            anchor="center",
        )

        # Backward compatibility placeholders
        self.adv_container = ctk.CTkFrame(self.scroll, fg_color=T.SURFACE, corner_radius=6)
        self.btn_toggle_adv = self.btn_open_adv

        # --- ACTION BUTTONS ---
        self.scroll.pack_forget()
        btn_box = ctk.CTkFrame(self, fg_color=T.PANEL, corner_radius=12, border_width=1, border_color=T.BORDER)
        btn_box.pack(side="bottom", fill="x", padx=4, pady=(8, 4))
        self.scroll.pack(fill="both", expand=True, padx=4, pady=4)

        self.btn_start = ctk.CTkButton(
            btn_box,
            text="START FL SERVER",
            height=44,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=T.PRIMARY,
            hover_color=T.PRIMARY_HOVER,
            text_color=T.ON_ACCENT,
            command=self._on_start_click,
        )
        self.btn_start.pack(fill="x", padx=12, pady=(10, 8))

        sub_btns = ctk.CTkFrame(btn_box, fg_color="transparent")
        sub_btns.pack(fill="x", padx=12, pady=(0, 10))

        self.btn_stop = ctk.CTkButton(
            sub_btns,
            text="STOP SERVER",
            height=34,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=T.ERROR,
            hover_color=T.DANGER_HOVER,
            text_color=T.ON_ACCENT,
            command=self._on_stop_click,
        )
        self.btn_stop.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.btn_export = ctk.CTkButton(
            sub_btns,
            text="Export Results",
            height=34,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=T.PANEL,
            hover_color=T.SURFACE,
            text_color=T.TEXT,
            border_width=1,
            border_color=T.BORDER,
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

    def _on_advanced_saved(self, new_cfg: Dict[str, Any], notify_parent: bool = True):
        """Persists updated advanced configuration and updates visual indicator."""
        self.advanced_config.update(new_cfg)
        overrides = 0
        for k, v in self.advanced_config.items():
            if k in DEFAULT_SERVER_ADVANCED and str(v) != str(DEFAULT_SERVER_ADVANCED[k]):
                overrides += 1
        if overrides > 0:
            self.lbl_adv_status.configure(
                text=f"[{overrides} Custom Overrides Active]",
                text_color=T.ACCENT,
            )
            self.btn_open_adv.configure(border_color=T.ACCENT)
        else:
            self.lbl_adv_status.configure(
                text="[Standard Defaults Active]",
                text_color=T.MUTED,
            )
            self.btn_open_adv.configure(border_color=T.BORDER)

        if notify_parent and getattr(self, "on_advanced_saved_cb", None):
            self.on_advanced_saved_cb(new_cfg)

    def _toggle_advanced(self):
        """Backward compatibility helper for headless test suites."""
        self.adv_expanded = not self.adv_expanded
        if self.adv_expanded:
            self.btn_toggle_adv.configure(text="Advanced Configuration (Active)")
        else:
            self.btn_toggle_adv.configure(text="Open Advanced Configuration...")

    def _on_strategy_changed(self, choice: str):
        pass

    def _on_run_mode_changed(self, choice: str):
        """Switch between Standalone and Simulation modes."""
        self.set_run_mode("simulation" if "simulation" in choice.lower() else "standalone")

    def set_run_mode(self, mode: str):
        """Show or hide simulation-specific controls and synchronize dropdown options and action buttons."""
        is_sim = "simulation" in str(mode).lower()
        self.run_mode = "simulation" if is_sim else "standalone"
        seg_label = "Simulation" if is_sim else "Standalone"
        if hasattr(self, "seg_run_mode") and self.seg_run_mode.get() != seg_label:
            self.seg_run_mode.set(seg_label)

        if is_sim:
            self.fields["dataset"].configure(values=DATASETS + ["All Datasets"])
            self.fields["strategy"].configure(values=STRATEGIES + ["All Strategies"])
            self.fields["partition_mode"].configure(values=PARTITION_MODES + ["All Partitions"])
            prev = self.field_rows["partition_mode"]
            for row_frame in self._sim_row_frames:
                row_frame.pack(fill="x", padx=10, pady=4, after=prev)
                prev = row_frame
        else:
            self.fields["dataset"].configure(values=DATASETS)
            if self.fields["dataset"].get() == "All Datasets":
                self.fields["dataset"].set(DATASETS[0])
            self.fields["strategy"].configure(values=STRATEGIES)
            if self.fields["strategy"].get() == "All Strategies":
                self.fields["strategy"].set(STRATEGIES[0])
            self.fields["partition_mode"].configure(values=PARTITION_MODES)
            if self.fields["partition_mode"].get() == "All Partitions":
                self.fields["partition_mode"].set(PARTITION_MODES[0])
            for row_frame in self._sim_row_frames:
                row_frame.pack_forget()

        self.set_running_state(self._is_running)
        if self.on_mode_change:
            self.on_mode_change(self.run_mode)

    @staticmethod
    def _append_opt_eval_common(cmd: List[str], adv: Dict[str, Any], optimizer: str, include_sim_extras: bool = False):
        """Append shared LR scheduling, optimizer, gradient, and early-stopping CLI flags."""
        lr_steps = str(adv.get("server_lr_decay_step", "")).strip()
        if lr_steps:
            cmd.extend(["--lr_decay_steps", lr_steps])
        lr_gamma = str(adv.get("server_lr_decay_gamma", "")).strip()
        if lr_gamma:
            cmd.extend(["--lr_decay_gamma", lr_gamma])
        s_lr_min = str(adv.get("server_lr_min", "")).strip()
        if s_lr_min:
            cmd.extend(["--server_lr_min", s_lr_min])
        c_pat = str(adv.get("client_lr_patience", "")).strip()
        if c_pat:
            cmd.extend(["--client_lr_patience", c_pat])
        c_fac = str(adv.get("client_lr_factor", "")).strip()
        if c_fac:
            cmd.extend(["--client_lr_factor", c_fac])
        c_min = str(adv.get("client_lr_min", "")).strip()
        if c_min:
            cmd.extend(["--client_lr_min", c_min])

        if optimizer == "sgd":
            mom = str(adv.get("momentum", "0.9")).strip()
            if mom:
                cmd.extend(["--momentum", mom])
        wd = str(adv.get("weight_decay", "")).strip()
        if wd:
            try:
                cmd.extend(["--weight_decay", wd])
            except ValueError:
                pass
        cgn = str(adv.get("clip_grad_norm", "")).strip()
        if cgn:
            try:
                if float(cgn) > 0:
                    cmd.extend(["--clip_grad_norm", cgn])
            except ValueError:
                pass

    def build_simulation_command(self, python_exe: str) -> List[str]:
        """Compile UI fields and advanced overrides into experiments.run_simulation CLI command."""
        ds_val = self.fields["dataset"].get()
        st_val = self.fields["strategy"].get()
        pt_val = self.fields["partition_mode"].get()
        opt = self.fields["optimizer"].get()

        cmd = [python_exe, "-u", "-m", "experiments.run_simulation"]
        cmd.extend(["--datasets", *(DATASETS if ds_val == "All Datasets" else [ds_val])])
        cmd.extend(["--strategies", *(STRATEGIES if st_val == "All Strategies" else [st_val])])
        cmd.extend(["--partitions", *(PARTITION_MODES if pt_val == "All Partitions" else [pt_val])])
        cmd.extend([
            "--model", self.fields["model"].get(),
            "--optimizer", opt,
            "--rounds", self.fields["rounds"].get().strip() or "40",
            "--num_clients", self.fields["min_clients"].get().strip() or "12",
            "--local_epochs", self.fields["local_epochs"].get().strip() or "5",
            "--batch_size", self.fields["batch_size"].get().strip() or "256",
            "--learning_rate", self.fields["learning_rate"].get().strip() or "0.01",
            "--lr_mode", self.fields["lr_mode"].get(),
            "--seed", self.fields["seed"].get().strip() or "42",
        ])

        run_name = self.fields["name"].get().strip()
        if run_name:
            cmd.extend(["--name", run_name])

        exec_val = self.fields["client_execution"].get()
        client_mod = "experiments.simulation_client" if "Sequential" in exec_val else "client.client"
        cmd.extend(["--client_module", client_mod])

        adv = self.advanced_config
        s_dir = adv.get("save_dir", "").strip()
        if s_dir:
            cmd.extend(["--output_dir", s_dir])

        strat_lower = st_val.lower()
        include_all_strats = (st_val == "All Strategies")
        if include_all_strats or "prox" in strat_lower:
            p_mu = str(adv.get("proximal_mu", "0.01")).strip()
            if p_mu:
                cmd.extend(["--proximal_mu", p_mu])
        if include_all_strats or "sol" in strat_lower:
            cmd.extend(["--fedsol_rho", str(adv.get("fedsol_rho", "2.0"))])
            cmd.extend(["--fedsol_target", str(adv.get("fedsol_target", "head"))])
            cmd.extend(["--fedsol_loss_type", str(adv.get("fedsol_loss_type", "kl"))])
            cmd.extend(["--fedsol_temperature", str(adv.get("fedsol_temperature", "3.0"))])
            if adv.get("fedsol_adaptive", False):
                cmd.append("--fedsol_adaptive")
        if include_all_strats or "switch" in strat_lower:
            sw_r = str(adv.get("switch_round", "0")).strip()
            try:
                if sw_r and int(sw_r) > 0:
                    cmd.extend(["--switch_round", sw_r])
            except ValueError:
                pass
            cmd.extend(["--fedswitch_weighting_mode", str(adv.get("fedswitch_weighting_mode", "bounded_sum"))])
            cmd.extend(["--fedswitch_coverage", str(adv.get("fedswitch_coverage", "0.9"))])
            cmd.extend(["--fedswitch_epsilon", str(adv.get("fedswitch_epsilon", "1.0"))])
            cmd.extend(["--fedswitch_c_bound", str(adv.get("fedswitch_c_bound", "50"))])
        if include_all_strats or "scr" in strat_lower:
            cmd.extend(["--scr_gamma", str(adv.get("scr_gamma", "1.0"))])
            cmd.extend(["--scr_head_lambda", str(adv.get("scr_head_lambda", "0.01"))])
        if include_all_strats or "sld" in strat_lower:
            cmd.extend(["--fedsld_epsilon", str(adv.get("fedsld_epsilon", "1e-6"))])
            cmd.extend(["--fedsld_reduction", str(adv.get("fedsld_reduction", "mean"))])

        self._append_opt_eval_common(cmd, adv, opt, include_sim_extras=True)

        cmd.append("--server_eval" if adv.get("server_eval", True) else "--no-server_eval")
        cmd.append("--local_test" if adv.get("local_test", False) else "--no-local_test")
        cmd.append("--augment" if adv.get("augment", False) else "--no-augment")

        es_p = str(adv.get("early_stop_patience", "0")).strip()
        try:
            if es_p and int(es_p) > 0:
                cmd.extend(["--early_stop_patience", es_p])
                cmd.extend(["--early_stop_metric", str(adv.get("early_stop_metric", "accuracy"))])
        except ValueError:
            pass

        n_wrk = str(adv.get("num_workers", "")).strip()
        if n_wrk:
            cmd.extend(["--num_workers", n_wrk])

        return cmd

    def build_command(self, python_exe: str, port: str = "8080") -> List[str]:
        """Compile parameters into exact server.server or experiments.run_simulation CLI command."""
        if self.run_mode == "simulation":
            return self.build_simulation_command(python_exe)

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

        # 0. Partition Mode
        pt_val = self.fields["partition_mode"].get() if "partition_mode" in self.fields else "practical"
        if pt_val and pt_val != "All Partitions":
            cmd.extend(["--partition_mode", pt_val])

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
            try:
                if sw_r and int(sw_r) > 0:
                    cmd.extend(["--switch_round", sw_r])
            except ValueError:
                pass
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
        self._append_opt_eval_common(cmd, adv, self.fields["optimizer"].get(), include_sim_extras=False)

        # 5. Evaluation & Early Stopping
        if not adv.get("server_eval", True):
            cmd.append("--no-server_eval")
        if adv.get("local_test", False):
            cmd.append("--local_test")
        else:
            cmd.append("--no-local_test")

        es_p = str(adv.get("early_stop_patience", "0")).strip()
        try:
            val_esp = int(es_p) if es_p else 0
            cmd.extend(["--early_stop_patience", str(val_esp)])
            if val_esp > 0:
                cmd.extend(["--early_stop_metric", str(adv.get("early_stop_metric", "accuracy"))])
        except ValueError:
            pass

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
        self._is_running = bool(running)
        start_txt = "START SIMULATION" if self.run_mode == "simulation" else "START FL SERVER"
        stop_txt = "STOP SIMULATION" if self.run_mode == "simulation" else "STOP SERVER"
        if running:
            self.btn_start.configure(text=start_txt, state="disabled", fg_color=T.BORDER)
            self.btn_stop.configure(text=stop_txt, state="normal", fg_color=T.DANGER)
        else:
            self.btn_start.configure(text=start_txt, state="normal", fg_color=T.PRIMARY)
            self.btn_stop.configure(text=stop_txt, state="disabled", fg_color=T.BORDER)
