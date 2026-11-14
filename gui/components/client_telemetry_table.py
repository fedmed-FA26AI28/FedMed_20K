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
        self.client_rows: Dict[int, Dict[str, ctk.CTkLabel]] = {}

        self._build_header()
        self._build_table_headers()
        self._build_scrollable_rows()
        self._build_footer()

    def _build_header(self):
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent", height=46)
        self.header_frame.pack(fill="x", padx=14, pady=(12, 6))

        # Title & Subtitle on Left
        title_box = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        title_box.pack(side="left", fill="y", anchor="w")

        self.lbl_title = ctk.CTkLabel(
            title_box,
            text="CLIENT TELEMETRY & ROUND STATS",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=T.TEXT,
        )
        self.lbl_title.pack(anchor="w")

        self.lbl_subtitle = ctk.CTkLabel(
            title_box,
            text=f"Reporting: 0/{self.total_clients} Nodes Connected | Synced Each Round from Clients",
            font=ctk.CTkFont(size=11),
            text_color=T.MUTED,
        )
        self.lbl_subtitle.pack(anchor="w")

        # Download CSV Button on Right (Plenty of horizontal space, never overlaps title)
        self.btn_export_csv = ctk.CTkButton(
            self.header_frame,
            text="Download Clients CSV",
            height=32,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=T.PRIMARY,
            hover_color=T.PRIMARY_HOVER,
            text_color=T.ON_ACCENT,
            command=self.export_csv,
        )
        self.btn_export_csv.pack(side="right", padx=(8, 0))

        # Divider
        divider = ctk.CTkFrame(self, height=1, fg_color=T.BORDER)
        divider.pack(fill="x", padx=14, pady=(4, 6))

    def _build_table_headers(self):
        """Header row with clean column alignment."""
        self.th_frame = ctk.CTkFrame(self, fg_color=T.SURFACE, height=30, corner_radius=4)
        self.th_frame.pack(fill="x", padx=14, pady=(0, 2))
        self.th_frame.pack_propagate(False)

        cols = [
            ("CLIENT", 0, 95, "w"),
            ("TRAIN LOSS", 1, 95, "w"),
            ("LOCAL ACC", 2, 95, "w"),
            ("TIME", 3, 75, "w"),
            ("PING", 4, 70, "w"),
            ("STATUS", 5, 80, "w"),
        ]

        for col_name, col_idx, width, anchor in cols:
            lbl = ctk.CTkLabel(
                self.th_frame,
                text=col_name,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=T.MUTED,
                width=width,
                anchor=anchor,
            )
            lbl.grid(row=0, column=col_idx, padx=6, pady=4, sticky="w")
            self.th_frame.grid_columnconfigure(col_idx, weight=1 if col_idx in (1, 2) else 0)

    def _build_scrollable_rows(self):
        self.scroll_frame = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            border_width=0,
        )
        self.scroll_frame.pack(fill="both", expand=True, padx=10, pady=(2, 6))
        for i in range(6):
            self.scroll_frame.grid_columnconfigure(i, weight=1 if i in (1, 2) else 0)

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

    def set_total_clients(self, count: int):
        self.total_clients = max(1, count)
        reporting = len(self.latest_client_states)
        self.lbl_subtitle.configure(
            text=f"Reporting: {reporting}/{self.total_clients} Nodes Connected | Synced Each Round from Clients"
        )

    def update_client_round(self, payload: Dict[str, Any]):
        """Receive round metrics for a client and update live row in real-time."""
        cid = int(payload.get("client_id", 0))
        self.latest_client_states[cid] = payload
        self.all_records.append(dict(payload))

        loss_val = float(payload.get("train_loss", 0.0))
        acc_val = float(payload.get("train_accuracy", 0.0))
        rtime_val = float(payload.get("round_time", payload.get("training_time", 0.0)))
        ping_val = float(payload.get("ping_ms", -1.0))

        # Check straggler condition (> 1.30x baseline of other peers, or median)
        other_times = [
            float(p.get("round_time", p.get("training_time", 0.0)))
            for c, p in self.latest_client_states.items()
            if c != cid and float(p.get("round_time", p.get("training_time", 0.0))) > 0
        ]
        base_time = (sum(other_times) / len(other_times)) if other_times else rtime_val
        is_slow = (rtime_val > base_time * 1.30 and rtime_val > 5.0)

        status_text = "SLOW" if is_slow else "DONE"
        status_color = T.WARNING if is_slow else T.SUCCESS

        if cid not in self.client_rows:
            row_idx = len(self.client_rows)
            row_widgets = {}

            lbl_cid = ctk.CTkLabel(
                self.scroll_frame,
                text=f"Client #{cid}",
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=T.TEXT,
                width=95,
                anchor="w",
            )
            lbl_cid.grid(row=row_idx, column=0, padx=6, pady=4, sticky="w")
            row_widgets["cid"] = lbl_cid

            lbl_loss = ctk.CTkLabel(
                self.scroll_frame,
                text=f"{loss_val:.4f}",
                font=ctk.CTkFont(family=T.MONO_FONT, size=11, weight="bold"),
                text_color=T.WARNING,
                width=95,
                anchor="w",
            )
            lbl_loss.grid(row=row_idx, column=1, padx=6, pady=4, sticky="w")
            row_widgets["loss"] = lbl_loss

            lbl_acc = ctk.CTkLabel(
                self.scroll_frame,
                text=f"{acc_val * 100:.2f}%",
                font=ctk.CTkFont(family=T.MONO_FONT, size=11, weight="bold"),
                text_color=T.SUCCESS,
                width=95,
                anchor="w",
            )
            lbl_acc.grid(row=row_idx, column=2, padx=6, pady=4, sticky="w")
            row_widgets["acc"] = lbl_acc

            lbl_time = ctk.CTkLabel(
                self.scroll_frame,
                text=f"{rtime_val:.2f}s",
                font=ctk.CTkFont(family=T.MONO_FONT, size=11),
                text_color=T.MUTED,
                width=75,
                anchor="w",
            )
            lbl_time.grid(row=row_idx, column=3, padx=6, pady=4, sticky="w")
            row_widgets["time"] = lbl_time

            ping_str = f"{ping_val:.0f}ms" if ping_val >= 0 else "—"
            lbl_ping = ctk.CTkLabel(
                self.scroll_frame,
                text=ping_str,
                font=ctk.CTkFont(family=T.MONO_FONT, size=11),
                text_color=T.BLUE,
                width=70,
                anchor="w",
            )
            lbl_ping.grid(row=row_idx, column=4, padx=6, pady=4, sticky="w")
            row_widgets["ping"] = lbl_ping

            lbl_status = ctk.CTkLabel(
                self.scroll_frame,
                text=status_text,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=status_color,
                width=80,
                anchor="w",
            )
            lbl_status.grid(row=row_idx, column=5, padx=6, pady=4, sticky="w")
            row_widgets["status"] = lbl_status

            self.client_rows[cid] = row_widgets
        else:
            row_widgets = self.client_rows[cid]
            row_widgets["loss"].configure(text=f"{loss_val:.4f}")
            row_widgets["acc"].configure(text=f"{acc_val * 100:.2f}%")
            row_widgets["time"].configure(text=f"{rtime_val:.2f}s")
            ping_str = f"{ping_val:.0f}ms" if ping_val >= 0 else "—"
            row_widgets["ping"].configure(text=ping_str)
            row_widgets["status"].configure(text=status_text, text_color=status_color)

        # Update header and footer
        reporting = len(self.latest_client_states)
        self.lbl_subtitle.configure(
            text=f"Reporting: {reporting}/{self.total_clients} Nodes Connected | Synced Each Round from Clients"
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
            self.lbl_diag.configure(
                text=f"Straggler Diagnostics: {', '.join(slow_clients)} reporting high latency",
                text_color=T.WARNING,
            )
        else:
            self.lbl_diag.configure(
                text="Straggler Diagnostics: All clients synchronizing within normal duration",
                text_color=T.SUCCESS,
            )

        pings = [float(p.get("ping_ms", -1.0)) for p in self.latest_client_states.values() if float(p.get("ping_ms", -1.0)) >= 0]
        avg_ping = sum(pings) / len(pings) if pings else 0.0
        self.lbl_net_stats.configure(
            text=f"Average Network Latency: {avg_ping:.1f} ms | Average Batch Duration: {avg_time:.2f} s"
        )

    def reset_table(self):
        """Clear all rows and reset telemetry table state."""
        self.latest_client_states.clear()
        self.all_records.clear()
        for widgets in self.client_rows.values():
            for w in widgets.values():
                w.destroy()
        self.client_rows.clear()
        self.lbl_subtitle.configure(
            text=f"Reporting: 0/{self.total_clients} Nodes Connected | Synced Each Round from Clients"
        )
        self.lbl_diag.configure(
            text="Straggler Diagnostics: Awaiting client node connections",
            text_color=T.MUTED,
        )
        self.lbl_net_stats.configure(
            text="Average Network Latency: — ms | Average Batch Duration: — s"
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
