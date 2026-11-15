import os
import csv
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from typing import Dict, Any, List, Optional
from gui import theme as T


class RecordedMetricsTable(ctk.CTkFrame):
    """Historical telemetry table showing recorded server and client metrics round-by-round."""

    def __init__(self, master, **kwargs):
        super().__init__(
            master,
            fg_color=T.PANEL,
            corner_radius=12,
            border_width=1,
            border_color=T.BORDER,
            **kwargs,
        )
        self.server_records: List[Dict[str, Any]] = []
        self.client_records: List[Dict[str, Any]] = []
        self.server_row_widgets: Dict[int, Dict[str, Any]] = {}
        self.client_row_widgets: Dict[Any, Dict[str, Any]] = {}
        self.active_view = "server"  # "server" or "client"
        self.client_filter = "All Clients"
        self.client_ids = set()
        self._last_counter_text: Optional[str] = None
        self._font_regular = ctk.CTkFont(size=10, weight="normal")
        self._font_bold = ctk.CTkFont(size=10, weight="bold")
        self._font_mono = ctk.CTkFont(family=T.MONO_FONT, size=10, weight="normal")
        self._font_mono_bold = ctk.CTkFont(family=T.MONO_FONT, size=10, weight="bold")

        self._build_ui()

    def _build_ui(self):
        # 1. Header Bar with Switcher & Export
        self.header_frame = ctk.CTkFrame(self, fg_color=T.SURFACE, height=40, corner_radius=6)
        self.header_frame.pack(fill="x", padx=8, pady=(8, 4))
        self.header_frame.pack_propagate(False)

        lbl_title = ctk.CTkLabel(
            self.header_frame,
            text="RECORDED METRICS",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=T.TEXT,
        )
        lbl_title.pack(side="left", padx=(10, 8))

        # View Switcher (Server vs Client)
        self.seg_view = ctk.CTkSegmentedButton(
            self.header_frame,
            values=["Server Global", "Client Nodes"],
            height=28,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=T.PANEL,
            selected_color=T.PRIMARY,
            selected_hover_color=T.PRIMARY_HOVER,
            unselected_color=T.PANEL,
            unselected_hover_color=T.SURFACE,
            text_color=T.TEXT,
            command=self._on_view_changed,
        )
        self.seg_view.set("Server Global")
        self.seg_view.pack(side="left", padx=4)

        # Client Filter Dropdown (shown only when in Client Nodes view)
        self.cmb_client_filter = ctk.CTkComboBox(
            self.header_frame,
            values=["All Clients"],
            width=125,
            height=28,
            font=ctk.CTkFont(size=11),
            fg_color=T.PANEL,
            border_color=T.BORDER,
            text_color=T.TEXT,
            button_color=T.PRIMARY,
            button_hover_color=T.PRIMARY_HOVER,
            command=self._on_client_filter_changed,
        )

        # Export CSV Button on Right
        self.btn_export = ctk.CTkButton(
            self.header_frame,
            text="Export CSV",
            width=88,
            height=28,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=T.PRIMARY,
            hover_color=T.PRIMARY_HOVER,
            text_color=T.ON_ACCENT,
            command=self.export_csv,
        )
        self.btn_export.pack(side="right", padx=(4, 8))

        # Subtitle counter on Right
        self.lbl_counter = ctk.CTkLabel(
            self.header_frame,
            text="0 Server Rounds",
            font=ctk.CTkFont(size=11),
            text_color=T.MUTED,
        )
        self.lbl_counter.pack(side="right", padx=8)

        # 2. Table Column Headers
        self.th_frame = ctk.CTkFrame(self, fg_color=T.PANEL, height=28, corner_radius=4)
        self.th_frame.pack(fill="x", padx=8, pady=(2, 2))
        self.th_frame.pack_propagate(False)

        # 3. Scrollable Rows Area
        self.scroll_frame = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            border_width=0,
        )
        self.scroll_frame.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        self._build_table_headers()

    def set_view_mode(self, mode: str):
        """Programmatically switch the active table view between 'server' and 'client'."""
        target = "server" if mode.lower() == "server" else "client"
        label = "Server Global" if target == "server" else "Client Nodes"
        self.seg_view.set(label)
        if target != self.active_view:
            self._on_view_changed(label)

    def _on_view_changed(self, choice: str):
        self.active_view = "server" if "Server" in choice else "client"
        if self.active_view == "client":
            self.cmb_client_filter.pack(side="left", padx=4, after=self.seg_view)
        else:
            self.cmb_client_filter.pack_forget()
        self._refresh_table()

    def _on_client_filter_changed(self, choice: str):
        self.client_filter = choice
        self._refresh_table()

    def _build_table_headers(self):
        for w in self.th_frame.winfo_children():
            w.destroy()

        if self.active_view == "server":
            cols = [
                ("ROUND", 0, 55, "w"),
                ("LOSS", 1, 75, "w"),
                ("BTA ACC", 2, 85, "w"),
                ("BMCTA ACC", 3, 90, "w"),
                ("TA", 4, 80, "w"),
                ("MCTA", 5, 75, "w"),
            ]
            uniform_grp = "srv_cols"
        else:
            cols = [
                ("RND", 0, 45, "w"),
                ("CLIENT", 1, 75, "w"),
                ("TR. LOSS", 2, 75, "w"),
                ("LOC. ACC", 3, 80, "w"),
                ("TIME", 4, 65, "w"),
                ("PING", 5, 55, "w"),
                ("STATUS", 6, 65, "w"),
            ]
            uniform_grp = "clt_cols"

        for col_name, col_idx, width, anchor in cols:
            lbl = ctk.CTkLabel(
                self.th_frame,
                text=col_name,
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color=T.MUTED,
                width=width,
                anchor=anchor,
            )
            lbl.grid(row=0, column=col_idx, padx=6, pady=2, sticky="ew")
            self.th_frame.grid_columnconfigure(col_idx, weight=1, uniform=uniform_grp)

    def add_server_metrics(
        self,
        round_num: int,
        loss: Optional[float] = None,
        bta: Optional[float] = None,
        bmcta: Optional[float] = None,
        ta: Optional[float] = None,
        mcta: Optional[float] = None,
    ):
        """Insert or consolidate server round metrics entry keyed by round number."""
        r_num = int(round_num)
        updates = {
            "loss": float(loss) if loss is not None else None,
            "bta": float(bta) if bta is not None else None,
            "bmcta": float(bmcta) if bmcta is not None else None,
            "ta": float(ta) if ta is not None else None,
            "mcta": float(mcta) if mcta is not None else None,
        }
        existing = next((r for r in self.server_records if r.get("round") == r_num), None)
        if existing is not None:
            for k, v in updates.items():
                if v is not None:
                    existing[k] = v
            record = existing
        else:
            record = {"round": r_num, **updates}
            self.server_records.append(record)
            self.server_records.sort(key=lambda r: int(r.get("round", 0)))

        if self.active_view == "server":
            cnt_txt = f"{len(self.server_records)} Server Rounds"
            if cnt_txt != self._last_counter_text:
                self._last_counter_text = cnt_txt
                self.lbl_counter.configure(text=cnt_txt)
            if r_num in self.server_row_widgets:
                self._update_server_row_widgets(r_num, record)
            else:
                self._render_single_server_row(record, len(self.server_records) - 1)

    def add_client_metrics(self, payload: Dict[str, Any]):
        """Insert or consolidate client round telemetry report keyed by (round, client_id)."""
        r_val = int(payload.get("server_round", payload.get("round", 0)))
        cid = int(payload.get("client_id", 0))
        key = (r_val, cid)
        is_new_cid = cid not in self.client_ids
        self.client_ids.add(cid)

        existing = next(
            (
                r for r in self.client_records
                if int(r.get("server_round", r.get("round", 0))) == r_val and int(r.get("client_id", 0)) == cid
            ),
            None,
        )
        if existing is not None:
            for k, v in payload.items():
                if v is not None:
                    existing[k] = v
            record = existing
        else:
            record = dict(payload)
            self.client_records.append(record)

        # Update filter dropdown values only when a new client id appears
        if is_new_cid:
            c_vals = ["All Clients"] + [f"Client #{c}" for c in sorted(self.client_ids)]
            self.cmb_client_filter.configure(values=c_vals)

        if self.active_view == "client":
            filtered_count = sum(
                1 for r in self.client_records
                if self.client_filter == "All Clients" or self.client_filter == f"Client #{r.get('client_id')}"
            )
            cnt_txt = f"{filtered_count} Reports"
            if cnt_txt != self._last_counter_text:
                self._last_counter_text = cnt_txt
                self.lbl_counter.configure(text=cnt_txt)
            if self.client_filter == "All Clients" or self.client_filter == f"Client #{cid}":
                if key in self.client_row_widgets:
                    self._update_client_row_widgets(key, record)
                else:
                    self._render_single_client_row(record, len(self.client_row_widgets))

    def _refresh_table(self):
        """Re-render all rows matching active view and filter."""
        self.server_row_widgets.clear()
        self.client_row_widgets.clear()
        self._build_table_headers()
        for w in self.scroll_frame.winfo_children():
            w.destroy()

        if self.active_view == "server":
            cnt_txt = f"{len(self.server_records)} Server Rounds"
            self._last_counter_text = cnt_txt
            self.lbl_counter.configure(text=cnt_txt)
            for i, rec in enumerate(self.server_records):
                self._render_single_server_row(rec, i)
        else:
            filtered = [
                rec for rec in self.client_records
                if self.client_filter == "All Clients" or self.client_filter == f"Client #{rec.get('client_id')}"
            ]
            cnt_txt = f"{len(filtered)} Reports"
            self._last_counter_text = cnt_txt
            self.lbl_counter.configure(text=cnt_txt)
            for i, rec in enumerate(filtered):
                self._render_single_client_row(rec, i)

    @staticmethod
    def _server_cell_specs(rec: Dict[str, Any]):
        return [
            (f"R{rec.get('round', 0)}", 55, T.TEXT, False),
            (f"{rec.get('loss'):.4f}" if rec.get("loss") is not None else "—", 75, T.WARNING, True),
            (f"{rec.get('bta') * 100:.2f}%" if rec.get("bta") is not None else "—", 85, T.SUCCESS, True),
            (f"{rec.get('bmcta') * 100:.2f}%" if rec.get("bmcta") is not None else "—", 90, T.BLUE, True),
            (f"{rec.get('ta') * 100:.2f}%" if rec.get("ta") is not None else "—", 80, T.ACCENT, True),
            (f"{rec.get('mcta') * 100:.2f}%" if rec.get("mcta") is not None else "—", 75, T.PURPLE, True),
        ]

    def _cell_font(self, is_mono: bool, is_bold: bool) -> ctk.CTkFont:
        if is_mono:
            return self._font_mono_bold if is_bold else self._font_mono
        return self._font_bold if is_bold else self._font_regular

    def _render_single_server_row(self, rec: Dict[str, Any], row_idx: int):
        r_num = int(rec.get("round", 0))
        r_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent", height=26)
        r_frame.pack(fill="x", pady=1)

        labels = []
        specs = self._server_cell_specs(rec)
        for col_idx, (text, width, color, is_mono) in enumerate(specs):
            font = self._cell_font(is_mono, col_idx in (2, 3))
            lbl = ctk.CTkLabel(r_frame, text=text, font=font, text_color=color, width=width, anchor="w")
            lbl._last_cell = (text, color)
            lbl.grid(row=0, column=col_idx, padx=6, pady=1, sticky="ew")
            r_frame.grid_columnconfigure(col_idx, weight=1, uniform="srv_cols")
            labels.append(lbl)

        self.server_row_widgets[r_num] = {"frame": r_frame, "labels": labels}

    def _update_server_row_widgets(self, r_num: int, rec: Dict[str, Any]):
        entry = self.server_row_widgets.get(r_num)
        if not entry:
            return
        for lbl, (text, _, color, _) in zip(entry["labels"], self._server_cell_specs(rec)):
            cell = (text, color)
            if getattr(lbl, "_last_cell", None) != cell:
                lbl.configure(text=text, text_color=color)
                lbl._last_cell = cell

    @staticmethod
    def _client_cell_specs(rec: Dict[str, Any]):
        r_val = rec.get("server_round", rec.get("round", 0))
        cid = rec.get("client_id", 0)
        t_loss = rec.get("train_loss", rec.get("loss"))
        t_acc = rec.get("train_accuracy", rec.get("ta"))
        r_time = rec.get("round_time", rec.get("training_time", 0.0))
        r_time_val = float(r_time) if r_time is not None else 0.0
        ping = rec.get("ping_ms", -1.0)
        ping_val = float(ping) if ping is not None else -1.0
        is_slow = r_time_val > 8.5  # Straggler threshold

        return [
            (f"R{r_val}", 45, T.MUTED, False),
            (f"Client #{cid}", 75, T.TEXT, False),
            (f"{float(t_loss):.4f}" if t_loss is not None else "—", 75, T.WARNING, True),
            (f"{float(t_acc) * 100:.2f}%" if t_acc is not None else "—", 80, T.SUCCESS, True),
            (f"{r_time_val:.2f}s", 65, T.MUTED, True),
            (f"{ping_val:.0f}ms" if ping_val >= 0 else "—", 55, T.BLUE, True),
            ("SLOW" if is_slow else "DONE", 65, T.WARNING if is_slow else T.SUCCESS, False),
        ]

    def _render_single_client_row(self, rec: Dict[str, Any], row_idx: int):
        r_val = int(rec.get("server_round", rec.get("round", 0)))
        cid = int(rec.get("client_id", 0))
        key = (r_val, cid)
        r_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent", height=26)
        r_frame.pack(fill="x", pady=1)

        labels = []
        for col_idx, (text, width, color, is_mono) in enumerate(self._client_cell_specs(rec)):
            font = self._cell_font(is_mono, col_idx in (3, 6))
            lbl = ctk.CTkLabel(r_frame, text=text, font=font, text_color=color, width=width, anchor="w")
            lbl._last_cell = (text, color)
            lbl.grid(row=0, column=col_idx, padx=6, pady=1, sticky="ew")
            r_frame.grid_columnconfigure(col_idx, weight=1, uniform="clt_cols")
            labels.append(lbl)

        self.client_row_widgets[key] = {"frame": r_frame, "labels": labels}

    def _update_client_row_widgets(self, key: Any, rec: Dict[str, Any]):
        entry = self.client_row_widgets.get(key)
        if not entry:
            return
        for lbl, (text, _, color, _) in zip(entry["labels"], self._client_cell_specs(rec)):
            cell = (text, color)
            if getattr(lbl, "_last_cell", None) != cell:
                lbl.configure(text=text, text_color=color)
                lbl._last_cell = cell

    def export_csv(self, filepath: Optional[str] = None, show_dialog: bool = True):
        """Export current table records to CSV."""
        if self.active_view == "server":
            records = self.server_records
            default_fn = "server_metrics_history.csv"
            fields = ["round", "loss", "bta", "bmcta", "ta", "mcta"]
        else:
            records = self.client_records
            default_fn = "client_metrics_history.csv"
            fields = [
                "server_round",
                "client_id",
                "train_loss",
                "train_accuracy",
                "eval_loss",
                "eval_accuracy",
                "round_time",
                "ping_ms",
                "num_samples",
                "device_ipv4",
            ]

        if not records:
            if show_dialog:
                messagebox.showinfo("Export CSV", f"No {self.active_view} records to export yet.")
            return

        if not filepath:
            os.makedirs("results", exist_ok=True)
            filepath = filedialog.asksaveasfilename(
                title=f"Export {self.active_view.capitalize()} Metrics CSV",
                initialdir="results",
                initialfile=default_fn,
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                defaultextension=".csv",
            )
            if not filepath:
                return

        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(records)

            if show_dialog:
                messagebox.showinfo(
                    "Export Complete",
                    f"Successfully exported {len(records)} records to:\n{filepath}",
                )
        except Exception as e:
            print(f"[EXPORT CSV ERROR] {e}")
            if show_dialog:
                messagebox.showerror("Export Error", f"Failed to export metrics CSV:\n{e}")

    def clear(self):
        """Clear all historical records and rows."""
        self.server_records.clear()
        self.client_records.clear()
        self.server_row_widgets.clear()
        self.client_row_widgets.clear()
        self.client_ids.clear()
        self.cmb_client_filter.configure(values=["All Clients"])
        self.cmb_client_filter.set("All Clients")
        self.client_filter = "All Clients"
        for w in self.scroll_frame.winfo_children():
            w.destroy()
        self.lbl_counter.configure(text="0 Records")
