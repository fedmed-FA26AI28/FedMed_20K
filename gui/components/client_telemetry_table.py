import os
import csv
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from typing import Dict, Any, List, Optional
from gui import theme as T


class ClientTelemetryTable(ctk.CTkFrame):
    """Real-time Client Connection and Round Telemetry Table for FL Server Control."""

    def __init__(self, master, total_clients: int = 12, **kwargs):
        super().__init__(
            master,
            fg_color=T.PANEL,
            corner_radius=8,
            border_width=1,
            border_color=T.BORDER,
            **kwargs,
        )
        self.total_clients = total_clients
        self.latest_client_states: Dict[int, Dict[str, Any]] = {}
        self.all_records: List[Dict[str, Any]] = []
        self.client_rows: Dict[Any, Dict[str, ctk.CTkLabel]] = {}
        self.round_filter: str = "Latest Round"
        self.recorded_rounds: set = set()
        self._font_bold = ctk.CTkFont(size=11, weight="bold")
        self._font_mono = ctk.CTkFont(family=T.MONO_FONT, size=11, weight="normal")
        self._font_mono_bold = ctk.CTkFont(family=T.MONO_FONT, size=11, weight="bold")

        self._build_header()
        self._build_table_headers()
        self._build_scrollable_rows()
        self._build_footer()

    def _build_header(self):
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.pack(fill="x", padx=14, pady=(10, 4))

        # Title & Subtitle on Left
        title_box = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        title_box.pack(side="left", fill="both", expand=True, anchor="w")

        self.lbl_title = ctk.CTkLabel(
            title_box,
            text="CLIENT TELEMETRY & ROUND STATS",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=T.TEXT,
            anchor="w",
        )
        self.lbl_title.pack(anchor="w")

        self.lbl_subtitle = ctk.CTkLabel(
            title_box,
            text=f"Reporting: 0/{self.total_clients} Nodes Connected | Synced Each Round from Clients",
            font=ctk.CTkFont(size=11),
            text_color=T.MUTED,
            anchor="w",
        )
        self.lbl_subtitle.pack(anchor="w")

        # Download CSV Button on Right
        self.btn_export_csv = ctk.CTkButton(
            self.header_frame,
            text="Download Clients CSV",
            height=30,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=T.PRIMARY,
            hover_color=T.PRIMARY_HOVER,
            text_color=T.ON_ACCENT,
            command=self.export_csv,
        )
        self.btn_export_csv.pack(side="right", padx=(8, 0))

        # Filter bar for Round selection (never crowds header on compact screens)
        filter_bar = ctk.CTkFrame(self, fg_color="transparent")
        filter_bar.pack(fill="x", padx=14, pady=(0, 4))

        lbl_rf = ctk.CTkLabel(
            filter_bar,
            text="Filter Round:",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=T.MUTED,
        )
        lbl_rf.pack(side="left", padx=(0, 6))

        self.cmb_round_filter = ctk.CTkComboBox(
            filter_bar,
            values=["Latest Round", "All Rounds"],
            width=130,
            height=26,
            font=ctk.CTkFont(size=11),
            fg_color=T.SURFACE,
            border_color=T.BORDER,
            text_color=T.TEXT,
            button_color=T.PRIMARY,
            button_hover_color=T.PRIMARY_HOVER,
            command=self._on_round_filter_changed,
        )
        self.cmb_round_filter.set("Latest Round")
        self.cmb_round_filter.pack(side="left")

        # Divider
        divider = ctk.CTkFrame(self, height=1, fg_color=T.BORDER)
        divider.pack(fill="x", padx=14, pady=(2, 6))

    def _build_table_headers(self):
        """Header row with clean column alignment."""
        self.th_frame = ctk.CTkFrame(self, fg_color=T.SURFACE, height=30, corner_radius=4)
        self.th_frame.pack(fill="x", padx=14, pady=(0, 2))
        self.th_frame.pack_propagate(False)

        cols = [
            ("ROUND", 0, 52, "w"),
            ("CLIENT", 1, 78, "w"),
            ("TRAIN LOSS", 2, 84, "w"),
            ("LOCAL ACC", 3, 84, "w"),
            ("TIME", 4, 65, "w"),
            ("PING", 5, 60, "w"),
            ("STATUS", 6, 65, "w"),
        ]

        for col_name, col_idx, width, anchor in cols:
            lbl = ctk.CTkLabel(
                self.th_frame,
                text=col_name,
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color=T.MUTED,
                width=width,
                anchor=anchor,
            )
            lbl.grid(row=0, column=col_idx, padx=5, pady=4, sticky="ew")
            self.th_frame.grid_columnconfigure(col_idx, weight=1, uniform="clt_tele_cols")

    def _build_scrollable_rows(self):
        self.scroll_frame = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            border_width=0,
        )
        self.scroll_frame.pack(fill="both", expand=True, padx=10, pady=(2, 6))
        for i in range(7):
            self.scroll_frame.grid_columnconfigure(i, weight=1, uniform="clt_tele_cols")

    def _build_footer(self):
        self.footer_frame = ctk.CTkFrame(self, fg_color=T.SURFACE, corner_radius=6, border_width=1, border_color=T.BORDER)
        self.footer_frame.pack(fill="x", padx=14, pady=(4, 12))

        self.lbl_diag = ctk.CTkLabel(
            self.footer_frame,
            text="Straggler Diagnostics: All clients synchronizing normally",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=T.TEXT,
            anchor="w",
        )
        self.lbl_diag.pack(anchor="w", padx=12, pady=(6, 2))

        self.lbl_net_stats = ctk.CTkLabel(
            self.footer_frame,
            text="Average Network Latency: — ms | Average Batch Duration: — s",
            font=ctk.CTkFont(size=10),
            text_color=T.MUTED,
            anchor="w",
        )
        self.lbl_net_stats.pack(anchor="w", padx=12, pady=(0, 6))

    @staticmethod
    def _cfg_if_changed(widget, **kwargs):
        cache = getattr(widget, "_last_cfg", None)
        if cache is None:
            cache = {}
            setattr(widget, "_last_cfg", cache)
        changed = {k: v for k, v in kwargs.items() if k not in cache or cache[k] != v}
        if changed:
            widget.configure(**changed)
            cache.update(changed)

    def set_total_clients(self, count: int):
        self.total_clients = max(1, count)
        reporting = len(self.latest_client_states)
        self._cfg_if_changed(
            self.lbl_subtitle,
            text=f"Reporting: {reporting}/{self.total_clients} Nodes Connected | Synced Each Round from Clients",
        )

    def _on_round_filter_changed(self, choice: str):
        self.round_filter = choice
        self._refresh_visible_rows()

    def _sync_round_filter_options(self):
        vals = ["Latest Round", "All Rounds"] + [f"Round {r}" for r in sorted(self.recorded_rounds)]
        try:
            current_vals = list(self.cmb_round_filter.cget("values") or [])
        except Exception:
            current_vals = []
        if current_vals != vals:
            self.cmb_round_filter.configure(values=vals)

    def _extract_row_display(self, payload: Dict[str, Any]):
        cid = int(payload.get("client_id", 0))
        srv_r = payload.get("server_round", payload.get("round"))
        rnd_txt = f"R{int(srv_r)}" if srv_r is not None else "—"

        train_l = payload.get("train_loss")
        eval_l = payload.get("eval_loss")
        if train_l is not None:
            loss_val = float(train_l)
        elif eval_l is not None:
            loss_val = float(eval_l)
        else:
            loss_val = 0.0

        train_a = payload.get("train_accuracy")
        eval_a = payload.get("eval_accuracy")
        if train_a is not None:
            acc_val = float(train_a)
        elif eval_a is not None:
            acc_val = float(eval_a)
        else:
            acc_val = 0.0

        rtime_val = float(payload.get("round_time", payload.get("training_time", 0.0)))
        ping_val = float(payload.get("ping_ms", -1.0))

        other_times = [
            float(p.get("round_time", p.get("training_time", 0.0)))
            for c, p in self.latest_client_states.items()
            if c != cid and float(p.get("round_time", p.get("training_time", 0.0))) > 0
        ]
        base_time = (sum(other_times) / len(other_times)) if other_times else rtime_val
        is_slow = (rtime_val > base_time * 1.30 and rtime_val > 5.0)
        status_text = "SLOW" if is_slow else "DONE"
        status_color = T.WARNING if is_slow else T.SUCCESS

        return {
            "cid": cid,
            "rnd_txt": rnd_txt,
            "loss_txt": f"{loss_val:.4f}",
            "acc_txt": f"{acc_val * 100:.2f}%",
            "time_txt": f"{rtime_val:.2f}s",
            "ping_txt": f"{ping_val:.0f}ms" if ping_val >= 0 else "—",
            "status_txt": status_text,
            "status_color": status_color,
            "rtime_val": rtime_val,
        }

    def _create_row_widgets(self, row_idx: int, disp: Dict[str, Any]) -> Dict[str, ctk.CTkLabel]:
        row_widgets: Dict[str, ctk.CTkLabel] = {}
        specs = [
            ("round", disp["rnd_txt"], self._font_bold, T.ACCENT, 52),
            ("cid", f"Client #{disp['cid']}", self._font_bold, T.TEXT, 78),
            ("loss", disp["loss_txt"], self._font_mono_bold, T.WARNING, 84),
            ("acc", disp["acc_txt"], self._font_mono_bold, T.SUCCESS, 84),
            ("time", disp["time_txt"], self._font_mono, T.MUTED, 65),
            ("ping", disp["ping_txt"], self._font_mono, T.BLUE, 60),
            ("status", disp["status_txt"], self._font_bold, disp["status_color"], 65),
        ]
        for col_idx, (key, text, font, color, width) in enumerate(specs):
            lbl = ctk.CTkLabel(
                self.scroll_frame,
                text=text,
                font=font,
                text_color=color,
                width=width,
                anchor="w",
            )
            lbl._last_cfg = {"text": text, "text_color": color}
            lbl.grid(row=row_idx, column=col_idx, padx=5, pady=4, sticky="ew")
            row_widgets[key] = lbl
        return row_widgets

    def _update_row_widgets(self, row_widgets: Dict[str, ctk.CTkLabel], disp: Dict[str, Any]):
        cfg = self._cfg_if_changed
        cfg(row_widgets["round"], text=disp["rnd_txt"])
        cfg(row_widgets["cid"], text=f"Client #{disp['cid']}")
        cfg(row_widgets["loss"], text=disp["loss_txt"])
        cfg(row_widgets["acc"], text=disp["acc_txt"])
        cfg(row_widgets["time"], text=disp["time_txt"])
        cfg(row_widgets["ping"], text=disp["ping_txt"])
        cfg(row_widgets["status"], text=disp["status_txt"], text_color=disp["status_color"])

    def _refresh_visible_rows(self):
        for w in self.scroll_frame.winfo_children():
            w.destroy()
        self.client_rows.clear()

        if self.round_filter == "Latest Round":
            records = [self.latest_client_states[c] for c in sorted(self.latest_client_states.keys())]
            key_fn = lambda rec: int(rec.get("client_id", 0))
        elif self.round_filter == "All Rounds":
            records = sorted(
                self.all_records,
                key=lambda r: (int(r.get("server_round", r.get("round", 0)) or 0), int(r.get("client_id", 0))),
            )
            key_fn = lambda rec: (int(rec.get("server_round", rec.get("round", 0)) or 0), int(rec.get("client_id", 0)))
        else:
            try:
                target_r = int(self.round_filter.split()[-1])
            except Exception:
                target_r = None
            records = sorted(
                [
                    r for r in self.all_records
                    if target_r is None or int(r.get("server_round", r.get("round", -1)) or -1) == target_r
                ],
                key=lambda r: int(r.get("client_id", 0)),
            )
            key_fn = lambda rec: int(rec.get("client_id", 0))

        for idx, rec in enumerate(records):
            disp = self._extract_row_display(rec)
            self.client_rows[key_fn(rec)] = self._create_row_widgets(idx, disp)

    def update_client_round(self, payload: Dict[str, Any]):
        """Receive round metrics for a client and update live row in real-time."""
        cid = int(payload.get("client_id", 0))
        merged_state = dict(self.latest_client_states.get(cid, {}))
        for k, v in payload.items():
            if v is None:
                continue
            if k in ("round_time", "training_time"):
                old_rt = merged_state.get(k)
                if float(v) <= 0.0 and old_rt is not None and float(old_rt) > 0.0:
                    continue
            if k == "ping_ms":
                old_ping = merged_state.get(k)
                is_eval_only = ("train_loss" not in payload and "eval_loss" in payload)
                if old_ping is not None and float(old_ping) >= 0.0 and (float(v) < 0.0 or is_eval_only):
                    continue
            merged_state[k] = v
        self.latest_client_states[cid] = merged_state

        srv_r = payload.get("server_round", payload.get("round"))
        if srv_r is not None:
            r_int = int(srv_r)
            self.recorded_rounds.add(r_int)
            self._sync_round_filter_options()
            existing_rec = next(
                (
                    rec for rec in self.all_records
                    if rec.get("server_round") is not None
                    and rec.get("client_id") is not None
                    and int(rec["server_round"]) == r_int
                    and int(rec["client_id"]) == cid
                ),
                None,
            )
            if existing_rec is not None:
                for k, v in payload.items():
                    if v is None:
                        continue
                    if k in ("round_time", "training_time"):
                        old_rt = existing_rec.get(k)
                        if float(v) <= 0.0 and old_rt is not None and float(old_rt) > 0.0:
                            continue
                    if k == "ping_ms":
                        old_ping = existing_rec.get(k)
                        is_eval_only = ("train_loss" not in payload and "eval_loss" in payload)
                        if old_ping is not None and float(old_ping) >= 0.0 and (float(v) < 0.0 or is_eval_only):
                            continue
                    existing_rec[k] = v
                active_rec = existing_rec
            else:
                active_rec = {k: v for k, v in payload.items() if v is not None}
                self.all_records.append(active_rec)
        else:
            active_rec = {k: v for k, v in payload.items() if v is not None}
            self.all_records.append(active_rec)

        disp = self._extract_row_display(merged_state if self.round_filter == "Latest Round" else active_rec)
        rtime_val = disp["rtime_val"]

        if self.round_filter == "Latest Round":
            if cid not in self.client_rows:
                self.client_rows[cid] = self._create_row_widgets(len(self.client_rows), disp)
            else:
                self._update_row_widgets(self.client_rows[cid], disp)
        else:
            self._refresh_visible_rows()

        # Update header and footer
        reporting = len(self.latest_client_states)
        self._cfg_if_changed(
            self.lbl_subtitle,
            text=f"Reporting: {reporting}/{self.total_clients} Nodes Connected | Synced Each Round from Clients",
        )

        # Diagnostics footer
        all_times = [
            float(p.get("round_time", p.get("training_time", 0.0)))
            for p in self.latest_client_states.values()
            if float(p.get("round_time", 0.0)) > 0
        ]
        avg_time = (sum(all_times) / len(all_times)) if all_times else rtime_val

        slow_clients = [
            f"Client #{c} ({p.get('round_time', 0.0):.2f}s)"
            for c, p in self.latest_client_states.items()
            if float(p.get("round_time", 0.0)) > avg_time * 1.30 and float(p.get("round_time", 0.0)) > 5.0
        ]
        if slow_clients:
            self._cfg_if_changed(
                self.lbl_diag,
                text=f"Straggler Diagnostics: {', '.join(slow_clients)} reporting high latency",
                text_color=T.WARNING,
            )
        else:
            self._cfg_if_changed(
                self.lbl_diag,
                text="Straggler Diagnostics: All clients synchronizing within normal duration",
                text_color=T.SUCCESS,
            )

        pings = [float(p.get("ping_ms", -1.0)) for p in self.latest_client_states.values() if float(p.get("ping_ms", -1.0)) >= 0]
        avg_ping = sum(pings) / len(pings) if pings else 0.0
        self._cfg_if_changed(
            self.lbl_net_stats,
            text=f"Average Network Latency: {avg_ping:.1f} ms | Average Batch Duration: {avg_time:.2f} s",
        )

    def reset_table(self):
        """Clear all rows and reset telemetry table state."""
        self.latest_client_states.clear()
        self.all_records.clear()
        self.recorded_rounds.clear()
        self.round_filter = "Latest Round"
        self.cmb_round_filter.configure(values=["Latest Round", "All Rounds"])
        self.cmb_round_filter.set("Latest Round")
        for w in self.scroll_frame.winfo_children():
            w.destroy()
        self.client_rows.clear()
        self._cfg_if_changed(
            self.lbl_subtitle,
            text=f"Reporting: 0/{self.total_clients} Nodes Connected | Synced Each Round from Clients",
        )
        self._cfg_if_changed(
            self.lbl_diag,
            text="Straggler Diagnostics: Awaiting client node connections",
            text_color=T.MUTED,
        )
        self._cfg_if_changed(
            self.lbl_net_stats,
            text="Average Network Latency: — ms | Average Batch Duration: — s",
        )

    def export_csv(self, filepath: Optional[str] = None, show_dialog: bool = True):
        """Export all recorded client telemetry rows to a CSV file."""
        if not self.all_records:
            if show_dialog:
                messagebox.showinfo("Export CSV", "No client telemetry records collected yet to export.")
            return

        if not filepath:
            os.makedirs("results", exist_ok=True)
            filepath = filedialog.asksaveasfilename(
                title="Save Client Telemetry CSV",
                initialdir="results",
                initialfile="client_telemetry_metrics.csv",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                defaultextension=".csv",
            )
            if not filepath:
                return

        try:
            fields = [
                "server_round",
                "client_id",
                "train_loss",
                "train_accuracy",
                "eval_loss",
                "eval_accuracy",
                "round_time",
                "epoch_time_avg",
                "ping_ms",
                "weight_size_kb",
                "num_samples",
                "device_ipv4",
                "device_type",
            ]
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.all_records)

            if show_dialog:
                messagebox.showinfo(
                    "Export Complete",
                    f"Successfully exported {len(self.all_records)} client round records to:\n{filepath}",
                )
        except Exception as e:
            print(f"[EXPORT ERROR] {e}")
            if show_dialog:
                messagebox.showerror("Export Error", f"Failed to export client telemetry CSV:\n{e}")
