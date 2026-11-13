import copy
import tkinter as tk
import customtkinter as ctk
import threading
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List
from gui.config_manager import (
    PROJECT_ROOT,
    DATASETS,
    STRATEGIES,
    MODELS,
    OPTIMIZERS,
    PARTITION_MODES,
    DEVICE_TYPES,
    load_partition_distribution,
    check_partition_exists,
    generate_partition_subprocess,
    get_default_venv_path,
)
from gui.components.advanced_config_dialog import (
    DEFAULT_CLIENT_ADVANCED,
    AdvancedConfigDialog,
)


class ClientPartitionInspector(ctk.CTkFrame):
    """Mini Canvas inspector showing local client class distribution and total sample counts."""

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="#f8fafc", corner_radius=8, border_width=1, border_color="#e8edf2", height=185, **kwargs)
        self._cached_draw = None
        self._build_ui()

    def _build_ui(self):
        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=10, pady=(6, 2))

        # Top line: Title (Left) + Badge (Right)
        top_row = ctk.CTkFrame(hdr, fg_color="transparent")
        top_row.pack(fill="x")

        self.lbl_title = ctk.CTkLabel(
            top_row,
            text="📊 PARTITION INSPECTOR",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#162d43",
        )
        self.lbl_title.pack(side="left")

        self.lbl_samples = ctk.CTkLabel(
            top_row,
            text="Unconfirmed",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#0c8c80",
            fg_color="#e6f7f5",
            corner_radius=4,
            padx=8,
            pady=2,
        )
        self.lbl_samples.pack(side="right")

        # Second line: Detailed Client Scope / Partition State
        self.lbl_subtitle = ctk.CTkLabel(
            hdr,
            text="Client 0 / 12 Clients Total",
            font=ctk.CTkFont(size=10),
            text_color="#64748b",
            anchor="w",
        )
        self.lbl_subtitle.pack(fill="x", pady=(1, 2))

        self.canvas = tk.Canvas(self, bg="#ffffff", highlightthickness=0, height=130)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=(2, 6))
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_canvas_configure(self, event):
        if event.width > 20 and event.height > 20 and self._cached_draw:
            self._cached_draw()

    def _get_canvas_dims(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 20:
            parent_w = self.winfo_width()
            w = (parent_w - 20) if parent_w > 50 else 340
        if h <= 20:
            h = 130
        return w, h

    def show_unconfirmed(self, num_clients: int, dataset: str, partition_mode: str, client_id: int = 0, seed: int = 42):
        """Display standby/unconfirmed card without prematurely showing any classes plot."""
        self.lbl_title.configure(text="📊 PARTITION INSPECTOR")
        self.lbl_subtitle.configure(text=f"Client Node #{client_id}  ·  {num_clients} Clients Total")
        self.lbl_samples.configure(text="Unconfirmed", text_color="#64748b", fg_color="#f1f5f9")

        def _draw():
            self.canvas.delete("all")
            w, h = self._get_canvas_dims()
            self.canvas.create_text(
                w / 2, h / 2 - 16,
                text="📋 Partition Not Confirmed",
                fill="#162d43",
                font=("Segoe UI", 11, "bold"),
            )
            self.canvas.create_text(
                w / 2, h / 2 + 14,
                text=f"Target: {dataset.upper()} | {partition_mode} | {num_clients} clients | Seed {seed}\nClick [🔍 Confirm & Inspect Partition] to verify or generate.",
                fill="#78889b",
                justify="center",
                width=w - 24,
                font=("Segoe UI", 9),
            )
        self._cached_draw = _draw
        _draw()

    def show_missing(self, num_clients: int, dataset: str, partition_mode: str, filename: str):
        """Display missing partition notice."""
        self.lbl_title.configure(text="📊 PARTITION INSPECTOR")
        self.lbl_subtitle.configure(text=f"{num_clients} Clients Partition Missing")
        self.lbl_samples.configure(text="Missing", text_color="#ba4c4c", fg_color="#fef2f2")

        def _draw():
            self.canvas.delete("all")
            w, h = self._get_canvas_dims()
            self.canvas.create_text(
                w / 2, h / 2 - 16,
                text=f"⚠️ Partition for {num_clients} Clients Missing",
                fill="#ba4c4c",
                font=("Segoe UI", 10, "bold"),
            )
            self.canvas.create_text(
                w / 2, h / 2 + 14,
                text=f"Expected: {filename}\nClick [⚡ Generate Partition] below, or start client to auto-generate.",
                fill="#64748b",
                justify="center",
                width=w - 24,
                font=("Segoe UI", 9),
            )
        self._cached_draw = _draw
        _draw()

    def show_generating(self, num_clients: int, dataset: str, partition_mode: str):
        """Display in-progress generating state."""
        self.lbl_title.configure(text="📊 PARTITION INSPECTOR")
        self.lbl_subtitle.configure(text=f"Generating {num_clients}-Client Partition...")
        self.lbl_samples.configure(text="Generating...", text_color="#d79338", fg_color="#fefce8")

        def _draw():
            self.canvas.delete("all")
            w, h = self._get_canvas_dims()
            self.canvas.create_text(
                w / 2, h / 2 - 14,
                text=f"⏳ Generating {num_clients}-Client Partition...",
                fill="#d79338",
                font=("Segoe UI", 11, "bold"),
            )
            self.canvas.create_text(
                w / 2, h / 2 + 14,
                text=f"Running partitioner for {dataset} ({partition_mode}). Please wait...",
                fill="#64748b",
                justify="center",
                width=w - 24,
                font=("Segoe UI", 9),
            )
        self._cached_draw = _draw
        _draw()

    def update_distribution(
        self,
        dataset: str,
        partition_mode: str,
        client_id: int,
        custom_path: Optional[str] = None,
        num_clients: Optional[int] = None,
        seed: int = 42,
    ):
        data = load_partition_distribution(
            dataset=dataset,
            partition_mode=partition_mode,
            client_id=client_id,
            num_clients=num_clients,
            seed=seed,
            custom_path=custom_path,
        )
        samples = data.get("samples", 0)
        dist = data.get("distribution", {})
        client_exists = data.get("client_exists", True)
        file_name = data.get("file")
        total_cls = data.get("num_clients", 0)

        self.lbl_title.configure(text="📊 PARTITION INSPECTOR")
        self.lbl_subtitle.configure(text=f"Client Node #{client_id}  ·  {total_cls} Clients in Partition")

        def _draw():
            self.canvas.delete("all")
            w, h = self._get_canvas_dims()

            if not file_name:
                self.lbl_samples.configure(text="No Partition", text_color="#f87171", fg_color="#fef2f2")
                self.canvas.create_text(
                    w / 2, h / 2,
                    text="No partition file found for selection\nGenerate partitions or specify custom path in Advanced",
                    fill="#94a3b8",
                    justify="center",
                    width=w - 24,
                    font=("Segoe UI", 9),
                )
                return

            if not client_exists:
                self.lbl_samples.configure(text="Out of Range", text_color="#f87171", fg_color="#fef2f2")
                self.canvas.create_text(
                    w / 2, h / 2 - 14,
                    text=f"⚠️ Client {client_id} exceeds partition bounds",
                    fill="#f87171",
                    font=("Segoe UI", 10, "bold"),
                )
                self.canvas.create_text(
                    w / 2, h / 2 + 14,
                    text=f"'{file_name}' contains {total_cls} clients (IDs 0-{max(0, total_cls-1)})",
                    fill="#94a3b8",
                    justify="center",
                    width=w - 24,
                    font=("Segoe UI", 9),
                )
                return

            self.lbl_samples.configure(text=f"{samples:,} Samples", text_color="#0c8c80", fg_color="#e6f7f5")
            if not dist:
                self.canvas.create_text(
                    w / 2, h / 2,
                    text=f"Client {client_id} has 0 samples assigned",
                    fill="#64748b",
                    font=("Segoe UI", 10),
                )
                return

            classes = sorted(dist.keys(), key=lambda x: int(x) if x.isdigit() else x)
            num_classes = len(classes)
            if num_classes == 0:
                return

            max_val = max(dist.values()) if dist.values() else 1
            bar_w = max(8, min(24, int((w - 40) / (num_classes * 1.4))))
            spacing = int((w - 40) / num_classes)
            start_x = 25
            base_y = h - 22

            self.canvas.create_line(15, base_y, w - 15, base_y, fill="#e2e8f0", width=1)

            for i, c in enumerate(classes):
                val = dist[c]
                bar_h = int((val / max_val) * (h - 50)) if max_val > 0 else 0
                x1 = start_x + i * spacing
                x2 = x1 + bar_w
                y1 = base_y - bar_h
                y2 = base_y

                color = "#0c8c80" if val > (max_val * 0.4) else "#94a3b8"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
                if bar_h > 14:
                    self.canvas.create_text((x1 + x2) / 2, y1 - 7, text=str(val), fill="#475569", font=("Segoe UI", 8))
                self.canvas.create_text((x1 + x2) / 2, base_y + 10, text=f"C{c}", fill="#64748b", font=("Segoe UI", 8))

        self._cached_draw = _draw
        _draw()


class ClientTab(ctk.CTkFrame):
    """FL Client Node tab: client parameters, partition inspector, and controls."""

    def __init__(
        self,
        master,
        on_start: Optional[Callable[[List[str]], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
        on_clear: Optional[Callable[[], None]] = None,
        get_python_exe: Optional[Callable[[], str]] = None,
        **kwargs,
    ):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_clear = on_clear
        self.get_python_exe = get_python_exe

        self.advanced_config: Dict[str, Any] = copy.deepcopy(DEFAULT_CLIENT_ADVANCED)
        self.adv_expanded = False
        self.partition_confirmed = False
        self.is_generating_partition = False
        self._build_form()

    def _build_form(self):
        self.scroll = ctk.CTkScrollableFrame(self, fg_color="#ffffff", corner_radius=8, border_width=1, border_color="#e8edf2", width=380)
        self.scroll.pack(fill="both", expand=True, padx=4, pady=4)

        title = ctk.CTkLabel(
            self.scroll,
            text="⚙️ CLIENT NODE PARAMETERS",
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

            kwargs.setdefault("fg_color", "#ffffff")
            kwargs.setdefault("border_color", "#dce3eb")
            kwargs.setdefault("text_color", "#162d43")
            widget = widget_cls(row, **kwargs)
            widget.pack(side="right", fill="x", expand=True, padx=(0, 4))
            return widget

        # 1. Client ID (Numerical stepper + direct entry)
        row_cid = ctk.CTkFrame(self.scroll, fg_color="transparent", height=32)
        row_cid.pack(fill="x", padx=10, pady=4)
        row_cid.pack_propagate(False)

        lbl_cid = ctk.CTkLabel(
            row_cid,
            text="Client ID (Index):",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#475569",
            width=140,
            anchor="w",
        )
        lbl_cid.pack(side="left", padx=(4, 8))

        stepper_frame = ctk.CTkFrame(row_cid, fg_color="transparent")
        stepper_frame.pack(side="right", fill="x", expand=True, padx=(0, 4))

        btn_dec = ctk.CTkButton(
            stepper_frame,
            text="−",
            width=28,
            height=28,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#f1f5f9",
            hover_color="#e2e8f0",
            text_color="#162d43",
            border_width=1,
            border_color="#dfe6ed",
            command=self._decrement_client_id,
        )
        btn_dec.pack(side="left", padx=(0, 4))

        self.entry_client_id = ctk.CTkEntry(
            stepper_frame,
            height=28,
            font=ctk.CTkFont(family="monospace", size=11, weight="bold"),
            fg_color="#ffffff",
            border_color="#dce3eb",
            text_color="#162d43",
            justify="center",
        )
        self.entry_client_id.insert(0, "0")
        self.entry_client_id.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.entry_client_id.bind("<KeyRelease>", lambda e: self._on_client_id_changed())

        btn_inc = ctk.CTkButton(
            stepper_frame,
            text="+",
            width=28,
            height=28,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#f1f5f9",
            hover_color="#e2e8f0",
            text_color="#162d43",
            border_width=1,
            border_color="#dfe6ed",
            command=self._increment_client_id,
        )
        btn_inc.pack(side="left", padx=(0, 0))

        def _set_cid(v: str):
            self.entry_client_id.delete(0, "end")
            self.entry_client_id.insert(0, str(v))
            self._on_client_id_changed()

        self.entry_client_id.set = _set_cid
        self.fields["client_id"] = self.entry_client_id

        # 2. Total Clients in Run (num_clients parameter)
        row_num_c = ctk.CTkFrame(self.scroll, fg_color="transparent", height=32)
        row_num_c.pack(fill="x", padx=10, pady=4)
        row_num_c.pack_propagate(False)

        lbl_num_c = ctk.CTkLabel(
            row_num_c,
            text="Total Clients (num_clients):",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#475569",
            width=140,
            anchor="w",
        )
        lbl_num_c.pack(side="left", padx=(4, 8))

        stepper_num_c = ctk.CTkFrame(row_num_c, fg_color="transparent")
        stepper_num_c.pack(side="right", fill="x", expand=True, padx=(0, 4))

        btn_dec_nc = ctk.CTkButton(
            stepper_num_c,
            text="−",
            width=28,
            height=28,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#f1f5f9",
            hover_color="#e2e8f0",
            text_color="#162d43",
            border_width=1,
            border_color="#dfe6ed",
            command=self._decrement_num_clients,
        )
        btn_dec_nc.pack(side="left", padx=(0, 4))

        self.entry_num_clients = ctk.CTkEntry(
            stepper_num_c,
            height=28,
            font=ctk.CTkFont(family="monospace", size=11, weight="bold"),
            fg_color="#ffffff",
            border_color="#dce3eb",
            text_color="#162d43",
            justify="center",
        )
        self.entry_num_clients.insert(0, "12")
        self.entry_num_clients.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.entry_num_clients.bind("<KeyRelease>", lambda e: self._on_params_changed_unconfirm())

        btn_inc_nc = ctk.CTkButton(
            stepper_num_c,
            text="+",
            width=28,
            height=28,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#f1f5f9",
            hover_color="#e2e8f0",
            text_color="#162d43",
            border_width=1,
            border_color="#dfe6ed",
            command=self._increment_num_clients,
        )
        btn_inc_nc.pack(side="left", padx=(0, 0))

        def _set_num_clients(v: str):
            self.entry_num_clients.delete(0, "end")
            self.entry_num_clients.insert(0, str(v))
            self._on_params_changed_unconfirm()

        self.entry_num_clients.set = _set_num_clients
        self.fields["num_clients"] = self.entry_num_clients

        # 3. Partition Scheme
        self.fields["partition_mode"] = add_field(
            "Partition Scheme:",
            ctk.CTkComboBox,
            values=PARTITION_MODES,
            height=28,
            font=ctk.CTkFont(size=11),
            command=self._on_params_changed_unconfirm,
        )
        self.fields["partition_mode"].set("practical")

        # 4. Dataset
        self.fields["dataset"] = add_field(
            "Dataset:",
            ctk.CTkComboBox,
            values=DATASETS,
            height=28,
            font=ctk.CTkFont(size=11),
            command=self._on_params_changed_unconfirm,
        )
        self.fields["dataset"].set("pathmnist")

        # 5. Strategy
        self.fields["strategy"] = add_field(
            "Assigned Strategy:",
            ctk.CTkComboBox,
            values=STRATEGIES,
            height=28,
            font=ctk.CTkFont(size=11),
        )
        self.fields["strategy"].set("fedbns")

        # 6. Model Architecture
        self.fields["model"] = add_field(
            "Model Architecture:",
            ctk.CTkComboBox,
            values=MODELS,
            height=28,
            font=ctk.CTkFont(size=11),
        )
        self.fields["model"].set("baseline_bn")

        # 7. Device Type Override
        self.fields["device_type"] = add_field(
            "Device Type:",
            ctk.CTkComboBox,
            values=DEVICE_TYPES,
            height=28,
            font=ctk.CTkFont(size=11),
        )
        self.fields["device_type"].set("auto")

        # 8. Local Epochs
        self.fields["local_epochs"] = add_field("Local Epochs:", ctk.CTkEntry, height=28, font=ctk.CTkFont(size=11))
        self.fields["local_epochs"].insert(0, "5")

        # 9. Batch Size
        self.fields["batch_size"] = add_field("Batch Size:", ctk.CTkEntry, height=28, font=ctk.CTkFont(size=11))
        self.fields["batch_size"].insert(0, "256")

        # 10. Learning Rate
        self.fields["learning_rate"] = add_field("Client LR:", ctk.CTkEntry, height=28, font=ctk.CTkFont(size=11))
        self.fields["learning_rate"].insert(0, "0.01")

        # 11. Optimizer
        self.fields["optimizer"] = add_field(
            "Optimizer:",
            ctk.CTkComboBox,
            values=OPTIMIZERS,
            height=28,
            font=ctk.CTkFont(size=11),
        )
        self.fields["optimizer"].set("sgd")

        # 12. Random Seed
        self.fields["seed"] = add_field("Random Seed:", ctk.CTkEntry, height=28, font=ctk.CTkFont(size=11))
        self.fields["seed"].insert(0, "42")
        self.fields["seed"].bind("<KeyRelease>", lambda e: self._on_params_changed_unconfirm())

        # --- PARTITION INSPECTOR ---
        self.inspector = ClientPartitionInspector(self.scroll)
        self.inspector.pack(fill="x", padx=10, pady=(14, 4))

        # Partition Action Toolbar
        part_toolbar = ctk.CTkFrame(self.scroll, fg_color="transparent")
        part_toolbar.pack(fill="x", padx=10, pady=(0, 4))

        self.btn_confirm_partition = ctk.CTkButton(
            part_toolbar,
            text="🔍 Confirm & Inspect Partition",
            height=30,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#0c8c80",
            hover_color="#087b70",
            text_color="#ffffff",
            command=self._confirm_and_inspect_partition,
        )
        self.btn_confirm_partition.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.btn_gen_partition = ctk.CTkButton(
            part_toolbar,
            text="⚡ Generate Partition",
            height=30,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#d79338",
            hover_color="#b45309",
            text_color="#ffffff",
            command=self._generate_missing_partition,
        )
        self.btn_gen_partition.pack(side="right", fill="x", expand=True, padx=(4, 0))
        self.btn_gen_partition.configure(state="disabled", fg_color="#cbd5e1")

        self.lbl_partition_status = ctk.CTkLabel(
            self.scroll,
            text="● Partition unconfirmed. Click [Confirm & Inspect Partition] to verify.",
            font=ctk.CTkFont(size=10),
            text_color="#64748b",
            anchor="w",
        )
        self.lbl_partition_status.pack(fill="x", padx=14, pady=(2, 6))

        # Show initial unconfirmed standby state (plot is hidden until confirmed)
        self.inspector.show_unconfirmed(
            num_clients=12,
            dataset="pathmnist",
            partition_mode="practical",
            client_id=0,
        )

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
        self.btn_open_adv.pack(fill="x", padx=10, pady=(10, 4))

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
        self.adv_partition_path = ctk.CTkEntry(self.adv_container)

        # --- ACTION BUTTONS ---
        btn_box = ctk.CTkFrame(self.scroll, fg_color="transparent")
        btn_box.pack(fill="x", padx=10, pady=(18, 12))

        self.btn_start = ctk.CTkButton(
            btn_box,
            text="▶ START FL CLIENT",
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
            text="■ STOP CLIENT",
            height=34,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#ba4c4c",
            hover_color="#991b1b",
            text_color="#ffffff",
            command=self._on_stop_click,
        )
        self.btn_stop.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.btn_clear = ctk.CTkButton(
            sub_btns,
            text="🔄 Clear Plots",
            height=34,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#ffffff",
            hover_color="#f3f7fa",
            text_color="#42566b",
            border_width=1,
            border_color="#dfe6ed",
            command=self._on_clear_click,
        )
        self.btn_clear.pack(side="right", fill="x", expand=True, padx=(4, 0))

    def _open_advanced_dialog(self):
        """Opens the modular multi-card Advanced Configuration sub-window for Client."""
        strat = self.fields["strategy"].get()
        AdvancedConfigDialog(
            parent=self.winfo_toplevel(),
            mode="client",
            current_config=self.advanced_config,
            active_strategy=strat,
            on_save=self._on_advanced_saved,
        )

    def _on_advanced_saved(self, new_cfg: Dict[str, Any]):
        """Persists updated client advanced configuration and updates visual indicator."""
        self.advanced_config.update(new_cfg)
        overrides = 0
        for k, v in self.advanced_config.items():
            if k in DEFAULT_CLIENT_ADVANCED and str(v) != str(DEFAULT_CLIENT_ADVANCED[k]):
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

    def _increment_client_id(self):
        try:
            val = int(self.entry_client_id.get().strip())
        except ValueError:
            val = 0
        self.entry_client_id.delete(0, "end")
        self.entry_client_id.insert(0, str(val + 1))
        self._on_client_id_changed()

    def _decrement_client_id(self):
        try:
            val = int(self.entry_client_id.get().strip())
        except ValueError:
            val = 0
        new_val = max(0, val - 1)
        self.entry_client_id.delete(0, "end")
        self.entry_client_id.insert(0, str(new_val))
        self._on_client_id_changed()

    def _increment_num_clients(self):
        try:
            val = int(self.entry_num_clients.get().strip())
        except ValueError:
            val = 12
        self.entry_num_clients.delete(0, "end")
        self.entry_num_clients.insert(0, str(val + 1))
        self._on_params_changed_unconfirm()

    def _decrement_num_clients(self):
        try:
            val = int(self.entry_num_clients.get().strip())
        except ValueError:
            val = 12
        new_val = max(1, val - 1)
        self.entry_num_clients.delete(0, "end")
        self.entry_num_clients.insert(0, str(new_val))
        self._on_params_changed_unconfirm()

    def _browse_partition_file(self):
        from tkinter import filedialog
        init_dir = str(PROJECT_ROOT / "data" / "partitions")
        f = filedialog.askopenfilename(
            initialdir=init_dir,
            title="Select Partition JSON File",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )
        if f:
            self.adv_partition_path.delete(0, "end")
            self.adv_partition_path.insert(0, f)
            self._on_params_changed_unconfirm()

    def _on_params_changed_unconfirm(self, _=None):
        """Reset partition confirmation when key dataset/client count parameters are altered."""
        self.partition_confirmed = False
        try:
            cid = int(self.fields["client_id"].get().strip())
        except ValueError:
            cid = 0
        try:
            nc = int(self.fields["num_clients"].get().strip())
        except ValueError:
            nc = 12
        ds = self.fields["dataset"].get()
        pm = self.fields["partition_mode"].get()
        seed = int(self.fields["seed"].get().strip()) if self.fields["seed"].get().strip().isdigit() else 42

        self.inspector.show_unconfirmed(num_clients=nc, dataset=ds, partition_mode=pm, client_id=cid, seed=seed)
        self.lbl_partition_status.configure(
            text="● Partition unconfirmed. Click [Confirm & Inspect Partition] to verify.",
            text_color="#94a3b8",
        )
        self.btn_gen_partition.configure(state="disabled", fg_color="#374151")

    def _on_client_id_changed(self, _=None):
        try:
            cid = int(self.fields["client_id"].get().strip())
        except ValueError:
            cid = 0
        try:
            nc = int(self.fields["num_clients"].get().strip())
        except ValueError:
            nc = 12
        ds = self.fields["dataset"].get()
        pm = self.fields["partition_mode"].get()
        seed = int(self.fields["seed"].get().strip()) if self.fields["seed"].get().strip().isdigit() else 42
        custom_path = self.adv_partition_path.get().strip() if hasattr(self, "adv_partition_path") else None

        if self.partition_confirmed:
            self.inspector.update_distribution(
                dataset=ds,
                partition_mode=pm,
                client_id=cid,
                custom_path=custom_path or None,
                num_clients=nc,
                seed=seed,
            )
        else:
            self.inspector.show_unconfirmed(num_clients=nc, dataset=ds, partition_mode=pm, client_id=cid, seed=seed)

    def _confirm_and_inspect_partition(self):
        """Confirm client count and load the class distribution plot, or report missing partition."""
        try:
            cid = int(self.fields["client_id"].get().strip())
        except ValueError:
            cid = 0
        try:
            nc = int(self.fields["num_clients"].get().strip())
        except ValueError:
            nc = 12
        ds = self.fields["dataset"].get()
        pm = self.fields["partition_mode"].get()
        seed = int(self.fields["seed"].get().strip()) if self.fields["seed"].get().strip().isdigit() else 42
        custom_path = self.adv_partition_path.get().strip() if hasattr(self, "adv_partition_path") else None

        exists, p_file = check_partition_exists(ds, pm, nc, seed, custom_path or None)
        if exists:
            self.partition_confirmed = True
            self.inspector.update_distribution(
                dataset=ds,
                partition_mode=pm,
                client_id=cid,
                custom_path=custom_path or None,
                num_clients=nc,
                seed=seed,
            )
            self.btn_gen_partition.configure(state="disabled", fg_color="#374151")
            self.lbl_partition_status.configure(
                text=f"✓ Confirmed partition file: {p_file.name}",
                text_color="#34d399",
            )
        else:
            self.partition_confirmed = False
            self.inspector.show_missing(num_clients=nc, dataset=ds, partition_mode=pm, filename=p_file.name)
            self.btn_gen_partition.configure(
                state="normal",
                fg_color="#d97706",
                text=f"⚡ Generate {nc}-Client Partition",
            )
            self.lbl_partition_status.configure(
                text=f"⚠️ Missing partition for {nc} clients. Click [Generate Partition] below.",
                text_color="#f87171",
            )

    def _generate_missing_partition(self):
        """Trigger background partition creation when missing."""
        try:
            nc = int(self.fields["num_clients"].get().strip())
        except ValueError:
            nc = 12
        ds = self.fields["dataset"].get()
        pm = self.fields["partition_mode"].get()
        seed = int(self.fields["seed"].get().strip()) if self.fields["seed"].get().strip().isdigit() else 42
        self._trigger_partition_generation(dataset=ds, partition_mode=pm, num_clients=nc, seed=seed)

    def _trigger_partition_generation(self, dataset: str, partition_mode: str, num_clients: int, seed: int, on_complete_callback=None):
        if self.is_generating_partition:
            return
        self.is_generating_partition = True
        self.inspector.show_generating(num_clients=num_clients, dataset=dataset, partition_mode=partition_mode)
        self.btn_confirm_partition.configure(state="disabled")
        self.btn_gen_partition.configure(state="disabled")
        self.lbl_partition_status.configure(
            text=f"⏳ Generating {num_clients}-client partition in background...",
            text_color="#fde68a",
        )

        py_exe = self.get_python_exe() if self.get_python_exe else get_default_venv_path()

        def worker():
            ok, msg = generate_partition_subprocess(
                dataset=dataset,
                partition_mode=partition_mode,
                num_clients=num_clients,
                seed=seed,
                local_test=True,
                python_exe=py_exe,
            )
            def done():
                self.is_generating_partition = False
                self.btn_confirm_partition.configure(state="normal")
                if ok:
                    self.partition_confirmed = True
                    try:
                        cid = int(self.fields["client_id"].get().strip())
                    except ValueError:
                        cid = 0
                    custom_path = self.adv_partition_path.get().strip() if hasattr(self, "adv_partition_path") else None
                    self.inspector.update_distribution(
                        dataset=dataset,
                        partition_mode=partition_mode,
                        client_id=cid,
                        custom_path=custom_path or None,
                        num_clients=num_clients,
                        seed=seed,
                    )
                    self.btn_gen_partition.configure(state="disabled", fg_color="#374151")
                    self.lbl_partition_status.configure(
                        text=f"✓ Auto-generated & confirmed {num_clients}-client partition!",
                        text_color="#34d399",
                    )
                    if on_complete_callback:
                        on_complete_callback()
                else:
                    self.lbl_partition_status.configure(text=f"❌ {msg}", text_color="#f87171")
                    self.btn_gen_partition.configure(state="normal", fg_color="#d97706")
            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def build_command(self, python_exe: str, server_address: str = "127.0.0.1:8080") -> List[str]:
        """Compile parameters into exact client.client CLI invocation command."""
        cmd = [
            python_exe,
            "-m",
            "client.client",
            "--client_id", self.fields["client_id"].get().strip(),
            "--num_clients", self.fields["num_clients"].get().strip(),
            "--server_address", str(server_address),
            "--dataset", self.fields["dataset"].get(),
            "--strategy", self.fields["strategy"].get(),
            "--model", self.fields["model"].get(),
            "--partition_mode", self.fields["partition_mode"].get(),
            "--local_epochs", self.fields["local_epochs"].get().strip(),
            "--batch_size", self.fields["batch_size"].get().strip(),
            "--learning_rate", self.fields["learning_rate"].get().strip(),
            "--optimizer", self.fields["optimizer"].get(),
            "--seed", self.fields["seed"].get().strip(),
        ]
        dev_type = self.fields["device_type"].get()
        if dev_type != "auto":
            cmd.extend(["--device_type", dev_type])

        adv = self.advanced_config

        # 1. Partition Path Override
        part = adv.get("partition_path", "").strip()
        if part:
            cmd.extend(["--partition_path", part])

        # 2. Client Dir
        c_dir = adv.get("client_dir", "").strip()
        if c_dir:
            cmd.extend(["--client_dir", c_dir])

        # 3. Pin Memory
        if not adv.get("pin_memory", True):
            cmd.append("--no-pin_memory")

        # 4. Augmentation
        if adv.get("augment", False):
            cmd.append("--augment")
        else:
            cmd.append("--no-augment")

        # 5. Local Test
        if adv.get("local_test", False):
            cmd.append("--local_test")

        # 6. Momentum & Optimizer
        opt = self.fields["optimizer"].get()
        if opt == "sgd":
            mom = str(adv.get("momentum", "0.9")).strip()
            if mom:
                cmd.extend(["--momentum", mom])

        # 7. Strategy Hyperparameters (e.g. FedSOL)
        strat = self.fields["strategy"].get().lower()
        if "sol" in strat:
            cmd.extend(["--fedsol_rho", str(adv.get("fedsol_rho", "2.0"))])
            cmd.extend(["--fedsol_target", str(adv.get("fedsol_target", "head"))])
            cmd.extend(["--fedsol_loss_type", str(adv.get("fedsol_loss_type", "kl"))])
            cmd.extend(["--fedsol_temperature", str(adv.get("fedsol_temperature", "3.0"))])
            if adv.get("fedsol_adaptive", False):
                cmd.append("--fedsol_adaptive")

        return cmd

    def _on_start_click(self):
        if not self.on_start:
            return

        try:
            nc = int(self.fields["num_clients"].get().strip())
        except ValueError:
            nc = 12
        ds = self.fields["dataset"].get()
        pm = self.fields["partition_mode"].get()
        seed = int(self.fields["seed"].get().strip()) if self.fields["seed"].get().strip().isdigit() else 42
        custom_path = self.advanced_config.get("partition_path", "").strip() or None

        exists, p_file = check_partition_exists(ds, pm, nc, seed, custom_path or None)
        if exists:
            self._confirm_and_inspect_partition()
            self.on_start(self.build_command(python_exe=""))
        else:
            # Auto-generate partition before starting!
            self.lbl_partition_status.configure(
                text=f"⏳ Auto-generating {nc}-client partition before launch...",
                text_color="#fde68a",
            )
            self._trigger_partition_generation(
                dataset=ds,
                partition_mode=pm,
                num_clients=nc,
                seed=seed,
                on_complete_callback=lambda: self.on_start(self.build_command(python_exe="")),
            )

    def _on_stop_click(self):
        if self.on_stop:
            self.on_stop()

    def _on_clear_click(self):
        if self.on_clear:
            self.on_clear()

    def set_running_state(self, running: bool):
        if running:
            self.btn_start.configure(state="disabled", fg_color="#374151")
            self.btn_stop.configure(state="normal", fg_color="#dc2626")
        else:
            self.btn_start.configure(state="normal", fg_color="#059669")
            self.btn_stop.configure(state="disabled", fg_color="#374151")
