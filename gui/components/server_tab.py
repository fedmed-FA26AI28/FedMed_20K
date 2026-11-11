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


class ServerTab(ctk.CTkFrame):
    """FL Server Control tab: parameter forms, accordion, and action buttons."""

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

        self.adv_expanded = False
        self._build_form()

    def _build_form(self):
        """Builds scrollable configuration form on the left with rock-solid row alignment."""
        self.scroll = ctk.CTkScrollableFrame(self, fg_color="#1e293b", corner_radius=8, width=380)
        self.scroll.pack(fill="both", expand=True, padx=4, pady=4)

        title = ctk.CTkLabel(
            self.scroll,
            text="⚙️ SERVER PARAMETER CONFIGURATION",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#f1f5f9",
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
                text_color="#93c5fd",
                width=140,
                anchor="w",
            )
            lbl.pack(side="left", padx=(4, 8))

            # Crucial: widget master is row, NOT self.scroll!
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

        # --- ADVANCED ACCORDION ---
        self.btn_toggle_adv = ctk.CTkButton(
            self.scroll,
            text="▶ Show Advanced Hyperparameters",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#334155",
            hover_color="#475569",
            command=self._toggle_advanced,
        )
        self.btn_toggle_adv.pack(fill="x", padx=10, pady=(14, 6))

        self.adv_container = ctk.CTkFrame(self.scroll, fg_color="#0f172a", corner_radius=6)

        def add_adv_field(label_text: str, default_val: str):
            r = ctk.CTkFrame(self.adv_container, fg_color="transparent", height=30)
            r.pack(fill="x", padx=8, pady=3)
            r.pack_propagate(False)
            lbl = ctk.CTkLabel(r, text=label_text, font=ctk.CTkFont(size=10), text_color="#cbd5e1", width=140, anchor="w")
            lbl.pack(side="left", padx=4)
            entry = ctk.CTkEntry(r, height=24, font=ctk.CTkFont(size=10))
            entry.insert(0, default_val)
            entry.pack(side="right", fill="x", expand=True, padx=4)
            return entry

        self.adv_mu = add_adv_field("FedProx mu:", "0.01")
        self.adv_rho = add_adv_field("FedSOL rho:", "2.0")
        self.adv_switch = add_adv_field("Switch Round (0=auto):", "0")
        self.adv_scr_gamma = add_adv_field("SCR gamma:", "1.0")
        self.adv_es_patience = add_adv_field("Early Stop Patience:", "0")

        # Partition Path Override with Browse Button
        r_part = ctk.CTkFrame(self.adv_container, fg_color="transparent", height=30)
        r_part.pack(fill="x", padx=8, pady=3)
        r_part.pack_propagate(False)

        lbl_part = ctk.CTkLabel(
            r_part,
            text="Partition Path Override:",
            font=ctk.CTkFont(size=10),
            text_color="#cbd5e1",
            width=140,
            anchor="w",
        )
        lbl_part.pack(side="left", padx=4)

        self.adv_partition_path = ctk.CTkEntry(
            r_part,
            height=24,
            font=ctk.CTkFont(family="monospace", size=10),
            placeholder_text="data/partitions/... (Optional)",
        )
        self.adv_partition_path.pack(side="left", fill="x", expand=True, padx=(4, 2))

        btn_browse = ctk.CTkButton(
            r_part,
            text="📁 Browse",
            width=65,
            height=24,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#2563eb",
            hover_color="#1d4ed8",
            command=self._browse_partition_file,
        )
        btn_browse.pack(side="right", padx=(2, 4))

        # --- ACTION BUTTONS ---
        btn_box = ctk.CTkFrame(self.scroll, fg_color="transparent")
        btn_box.pack(fill="x", padx=10, pady=(18, 12))

        self.btn_start = ctk.CTkButton(
            btn_box,
            text="▶ START FEDERATED SERVER",
            height=46,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#059669",
            hover_color="#047857",
            command=self._on_start_click,
        )
        self.btn_start.pack(fill="x", pady=(0, 8))

        sub_btns = ctk.CTkFrame(btn_box, fg_color="transparent")
        sub_btns.pack(fill="x")

        self.btn_stop = ctk.CTkButton(
            sub_btns,
            text="■ STOP / KILL",
            height=36,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#dc2626",
            hover_color="#b91c1c",
            command=self._on_stop_click,
        )
        self.btn_stop.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.btn_export = ctk.CTkButton(
            sub_btns,
            text="📊 Export CSV",
            height=36,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#4f46e5",
            hover_color="#4338ca",
            command=self._on_export_click,
        )
        self.btn_export.pack(side="right", fill="x", expand=True, padx=(4, 0))

    def _toggle_advanced(self):
        self.adv_expanded = not self.adv_expanded
        if self.adv_expanded:
            self.adv_container.pack(fill="x", padx=10, pady=(0, 8))
            self.btn_toggle_adv.configure(text="▼ Hide Advanced Hyperparameters")
        else:
            self.adv_container.pack_forget()
            self.btn_toggle_adv.configure(text="▶ Show Advanced Hyperparameters")

    def _browse_partition_file(self):
        from tkinter import filedialog
        init_dir = str(PROJECT_ROOT / "data" / "partitions")
        f = filedialog.askopenfilename(
            initialdir=init_dir,
            title="Select Partition JSON File for Server",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )
        if f:
            self.adv_partition_path.delete(0, "end")
            self.adv_partition_path.insert(0, f)

    def _on_strategy_changed(self, choice: str):
        strat = choice.lower()
        if "prox" in strat:
            self.adv_mu.configure(fg_color="#1e3a8a")
        else:
            self.adv_mu.configure(fg_color="#1e293b")

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

        # Check partition path override
        if hasattr(self, "adv_partition_path"):
            part = self.adv_partition_path.get().strip()
            if part:
                cmd.extend(["--partition_path", part])

        # Advanced options
        if self.adv_expanded:
            p_mu = self.adv_mu.get().strip()
            if p_mu:
                cmd.extend(["--proximal_mu", p_mu])
            p_rho = self.adv_rho.get().strip()
            if p_rho:
                cmd.extend(["--fedsol_rho", p_rho])
            p_sw = self.adv_switch.get().strip()
            if p_sw and int(p_sw) > 0:
                cmd.extend(["--switch_round", p_sw])
            p_gamma = self.adv_scr_gamma.get().strip()
            if p_gamma:
                cmd.extend(["--scr_gamma", p_gamma])
            p_es = self.adv_es_patience.get().strip()
            if p_es and int(p_es) > 0:
                cmd.extend(["--early_stop_patience", p_es])

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
