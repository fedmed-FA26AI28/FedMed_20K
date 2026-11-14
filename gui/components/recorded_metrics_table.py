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
        self.active_view = "server"  # "server" or "client"
        self.client_filter = "All Clients"
        self.client_ids = set()

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

        for col_name, col_idx, width, anchor in cols:
            lbl = ctk.CTkLabel(
                self.th_frame,
                text=col_name,
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color=T.MUTED,
                width=width,
                anchor=anchor,
            )
            lbl.grid(row=0, column=col_idx, padx=4, pady=2, sticky="w")
            self.th_frame.grid_columnconfigure(col_idx, weight=1 if col_idx in (1, 2, 3) else 0)

    def add_server_metrics(
        self,
        round_num: int,
        loss: Optional[float] = None,
        bta: Optional[float] = None,
        bmcta: Optional[float] = None,
        ta: Optional[float] = None,
        mcta: Optional[float] = None,
    ):
        """Append a server round metrics entry."""
        record = {
            "round": int(round_num),
            "loss": float(loss) if loss is not None else None,
            "bta": float(bta) if bta is not None else None,
            "bmcta": float(bmcta) if bmcta is not None else None,
            "ta": float(ta) if ta is not None else None,
            "mcta": float(mcta) if mcta is not None else None,
        }
        self.server_records.append(record)

        if self.active_view == "server":
            self.lbl_counter.configure(text=f"{len(self.server_records)} Server Rounds")
            self._render_single_server_row(record, len(self.server_records) - 1)

    def add_client_metrics(self, payload: Dict[str, Any]):
        """Append a client round telemetry report."""
        record = dict(payload)
        cid = int(record.get("client_id", 0))
        self.client_ids.add(cid)
        self.client_records.append(record)

        # Update filter dropdown values
        c_vals = ["All Clients"] + [f"Client #{c}" for c in sorted(self.client_ids)]
        self.cmb_client_filter.configure(values=c_vals)

        if self.active_view == "client":
            self.lbl_counter.configure(text=f"{len(self.client_records)} Reports")
            if self.client_filter == "All Clients" or self.client_filter == f"Client #{cid}":
                self._render_single_client_row(record, len(self.scroll_frame.winfo_children()))

    def _refresh_table(self):
        """Re-render all rows matching active view and filter."""
        self._build_table_headers()
        for w in self.scroll_frame.winfo_children():
            w.destroy()

        if self.active_view == "server":
            self.lbl_counter.configure(text=f"{len(self.server_records)} Server Rounds")
            for i, rec in enumerate(self.server_records):
                self._render_single_server_row(rec, i)
        else:
            filtered = [
                rec for rec in self.client_records
                if self.client_filter == "All Clients" or self.client_filter == f"Client #{rec.get('client_id')}"
            ]
            self.lbl_counter.configure(text=f"{len(filtered)} Reports")
            for i, rec in enumerate(filtered):
                self._render_single_client_row(rec, i)

    def _render_single_server_row(self, rec: Dict[str, Any], row_idx: int):
        r_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent", height=26)
        r_frame.pack(fill="x", pady=1)

        cols = [
            (f"R{rec.get('round', 0)}", 55, T.TEXT, False),
            (f"{rec.get('loss'):.4f}" if rec.get("loss") is not None else "—", 75, T.WARNING, True),
            (f"{rec.get('bta') * 100:.2f}%" if rec.get("bta") is not None else "—", 85, T.SUCCESS, True),
            (f"{rec.get('bmcta') * 100:.2f}%" if rec.get("bmcta") is not None else "—", 90, T.BLUE, True),
            (f"{rec.get('ta') * 100:.2f}%" if rec.get("ta") is not None else "—", 80, T.ACCENT, True),
            (f"{rec.get('mcta') * 100:.2f}%" if rec.get("mcta") is not None else "—", 75, T.PURPLE, True),
        ]

        for col_idx, (text, width, color, is_mono) in enumerate(cols):
            font = ctk.CTkFont(family=T.MONO_FONT if is_mono else None, size=10, weight="bold" if col_idx in (2, 3) else "normal")
            lbl = ctk.CTkLabel(r_frame, text=text, font=font, text_color=color, width=width, anchor="w")
            lbl.grid(row=0, column=col_idx, padx=4, pady=1, sticky="w")
            r_frame.grid_columnconfigure(col_idx, weight=1 if col_idx in (1, 2, 3) else 0)

    def _render_single_client_row(self, rec: Dict[str, Any], row_idx: int):
        r_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent", height=26)
        r_frame.pack(fill="x", pady=1)

        r_val = rec.get("server_round", rec.get("round", 0))
        cid = rec.get("client_id", 0)
        t_loss = rec.get("train_loss", 0.0)
        t_acc = rec.get("train_accuracy", 0.0)
        r_time = rec.get("round_time", rec.get("training_time", 0.0))
        ping = rec.get("ping_ms", -1.0)
        is_slow = float(r_time) > 8.5  # Straggler threshold

        cols = [
            (f"R{r_val}", 45, T.MUTED, False),
            (f"Client #{cid}", 75, T.TEXT, False),
            (f"{float(t_loss):.4f}", 75, T.WARNING, True),
            (f"{float(t_acc) * 100:.2f}%", 80, T.SUCCESS, True),
            (f"{float(r_time):.2f}s", 65, T.MUTED, True),
            (f"{float(ping):.0f}ms" if float(ping) >= 0 else "—", 55, T.BLUE, True),
            ("SLOW" if is_slow else "DONE", 65, T.WARNING if is_slow else T.SUCCESS, False),
        ]

        for col_idx, (text, width, color, is_mono) in enumerate(cols):
            font = ctk.CTkFont(family=T.MONO_FONT if is_mono else None, size=10, weight="bold" if col_idx in (3, 6) else "normal")
            lbl = ctk.CTkLabel(r_frame, text=text, font=font, text_color=color, width=width, anchor="w")
            lbl.grid(row=0, column=col_idx, padx=4, pady=1, sticky="w")
            r_frame.grid_columnconfigure(col_idx, weight=1 if col_idx in (2, 3) else 0)

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
        self.client_ids.clear()
        self.cmb_client_filter.configure(values=["All Clients"])
        self.cmb_client_filter.set("All Clients")
        self.client_filter = "All Clients"
        for w in self.scroll_frame.winfo_children():
            w.destroy()
        self.lbl_counter.configure(text="0 Records")
