"""Utility to probe and log ping latency across all edge Jetson clients.

Usage:
    # Quick one-shot ping sweep to all 10 clients
    python -m scripts.ping_clients

    # Probe 5 times with 1-second interval and save to results/network/
    python -m scripts.ping_clients --count 5 --interval 1.0

    # Probe custom server/client IP
    python -m scripts.ping_clients --ips 192.168.1.50 192.168.1.51 127.0.0.1
"""

import os
import sys
import time
import argparse
import datetime
import csv
from pathlib import Path
from typing import List, Dict, Any

from monitoring.network import measure_ping
from scripts.distribute_data import JETSON_CLIENTS


def run_ping_sweep(
    clients: List[Dict[str, Any]],
    count: int = 1,
    interval: float = 1.0,
    check_port: int = 8080,
    save_dir: str = "results/network",
) -> List[Dict[str, Any]]:
    """Probe network latency to all client devices and record results."""
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / "clients_ping.log"
    csv_file = out_dir / "clients_ping.csv"

    write_header = not csv_file.exists()

    all_results = []

    print(f"\n{'='*75}")
    print(f"  FedMedAI Client Network Ping Probe (Target Port: {check_port})")
    print(f"  Probes per client: {count} | Interval: {interval}s")
    print(f"  Logging to: {csv_file}")
    print(f"{'='*75}")

    header = f"{'Client ID':<10} | {'Type':<13} | {'IP Address':<16} | {'Avg Ping (ms)':<14} | {'Status':<10}"
    print(header)
    print("-" * 75)

    for client in clients:
        cid = client.get("client_id", -1)
        dev_type = client.get("device_type", "Unknown")
        ip = client.get("ip", "127.0.0.1")

        pings = []
        statuses = []

        for p_idx in range(count):
            res = measure_ping(host=ip, port=check_port, timeout_seconds=1.0)
            if res["ping_ms"] is not None:
                pings.append(res["ping_ms"])
            statuses.append(res["status"])

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Append to CSV
            with open(csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "timestamp",
                        "client_id",
                        "device_type",
                        "ip",
                        "port",
                        "probe_index",
                        "ping_ms",
                        "icmp_ms",
                        "tcp_ms",
                        "status",
                    ])
                    write_header = False
                writer.writerow([
                    timestamp,
                    cid,
                    dev_type,
                    ip,
                    check_port,
                    p_idx + 1,
                    res["ping_ms"] if res["ping_ms"] is not None else "",
                    res["icmp_ms"] if res["icmp_ms"] is not None else "",
                    res["tcp_ms"] if res["tcp_ms"] is not None else "",
                    res["status"],
                ])

            if count > 1 and p_idx < count - 1:
                time.sleep(interval)

        import numpy as np
        avg_ping = f"{np.mean(pings):.2f}" if pings else "N/A"
        final_status = "ONLINE" if pings else (statuses[-1] if statuses else "OFFLINE")

        summary_entry = {
            "client_id": cid,
            "device_type": dev_type,
            "ip": ip,
            "avg_ping_ms": float(np.mean(pings)) if pings else None,
            "status": final_status,
        }
        all_results.append(summary_entry)

        # Print line
        print(f"Client {cid:<3} | {dev_type:<13} | {ip:<16} | {avg_ping:<14} | {final_status:<10}")

        # Append to human-readable log
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(
                f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"Client {cid} ({dev_type}) at {ip} | Avg Ping: {avg_ping} ms | Status: {final_status}\n"
            )

    print(f"{'='*75}\n")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="FedMedAI Client Ping Tool")
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of ping probes per client (default: 1)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Interval between probes in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Target port to probe (default: 8080)",
    )
    parser.add_argument(
        "--ips",
        nargs="+",
        default=None,
        help="Optional list of custom IP addresses to ping",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/network",
        help="Directory to save ping logs and CSVs",
    )
    args = parser.parse_args()

    if args.ips:
        client_list = [
            {"client_id": idx, "device_type": "Custom", "ip": ip}
            for idx, ip in enumerate(args.ips)
        ]
    else:
        client_list = JETSON_CLIENTS

    run_ping_sweep(
        clients=client_list,
        count=args.count,
        interval=args.interval,
        check_port=args.port,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
