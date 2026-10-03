"""Network latency profiling and client ping logging for FedMedAI.

Provides cross-platform ICMP and TCP ping measurement and persistent
per-client ping logs for edge and simulated federated learning setups.
"""

import os
import sys
import re
import csv
import time
import socket
import datetime
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union


def parse_server_address(server_address: str, default_port: int = 8080) -> Tuple[str, int]:
    """Parse server address string into (host, port).

    Supports formats:
        - "127.0.0.1:8080" -> ("127.0.0.1", 8080)
        - "192.168.1.50"   -> ("192.168.1.50", 8080)
        - "localhost:5000" -> ("localhost", 5000)
        - "http://10.0.0.1:8080" -> ("10.0.0.1", 8080)
        - "[::1]:8080"     -> ("::1", 8080)
    """
    addr = server_address.strip()
    # Strip scheme if present
    if "://" in addr:
        addr = addr.split("://", 1)[1]

    # Handle IPv6 bracket format [host]:port
    if addr.startswith("["):
        end_bracket = addr.find("]")
        if end_bracket != -1:
            host = addr[1:end_bracket]
            rest = addr[end_bracket + 1:]
            port = int(rest.lstrip(":")) if rest.startswith(":") else default_port
            return host, port

    # Handle host:port
    if ":" in addr:
        parts = addr.rsplit(":", 1)
        host = parts[0]
        try:
            port = int(parts[1])
        except ValueError:
            port = default_port
        return host, port

    return addr, default_port


def get_device_ipv4() -> str:
    """Return the primary local IPv4 address of this machine.

    Uses a zero-packet UDP socket lookup (standard library) with fallbacks
    to hostname resolution. Works across Windows and Linux (Jetson Orin/Nano).
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass

    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass

    return "127.0.0.1"


def measure_icmp_ping(host: str, timeout_seconds: float = 1.0) -> Optional[float]:
    """Measure round-trip ICMP ping latency in milliseconds.

    Works across Windows and Linux (including Jetson Orin/Nano).
    Returns None if unreachable, timed out, or blocked.
    """
    clean_host = "127.0.0.1" if host in ("0.0.0.0", "") else host

    is_win = sys.platform.startswith("win")
    if is_win:
        timeout_ms = max(100, int(timeout_seconds * 1000))
        cmd = ["ping", "-n", "1", "-w", str(timeout_ms), clean_host]
    else:
        timeout_sec = max(1, int(round(timeout_seconds)))
        cmd = ["ping", "-c", "1", "-W", str(timeout_sec), clean_host]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds + 1.5,
        )
        output = proc.stdout + proc.stderr

        # Check return code
        if proc.returncode != 0 and "time" not in output.lower():
            return None

        # Parse latency
        # 1. Matches "time<1ms" or "time=0.45ms" or "time=12 ms"
        m = re.search(r"time[=<]([0-9.]+)\s?ms", output, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            return val

        if "time<1ms" in output.lower():
            return 0.5

        # 2. Linux RTT stats: "rtt min/avg/max/mdev = 0.045/0.065/0.085/0.010 ms"
        m_rtt = re.search(r"rtt\s+min/avg/max/\w+\s*=\s*[0-9.]+/([0-9.]+)/", output, re.IGNORECASE)
        if m_rtt:
            return float(m_rtt.group(1))

        # 3. Windows stats summary: "Average = 2ms"
        m_win = re.search(r"Average\s*=\s*([0-9]+)\s*ms", output, re.IGNORECASE)
        if m_win:
            return float(m_win.group(1))

    except Exception:
        pass

    return None


def measure_tcp_ping(host: str, port: int, timeout_seconds: float = 1.0) -> Tuple[Optional[float], str]:
    """Measure TCP socket connection handshake latency in milliseconds.

    Returns:
        Tuple of (latency_ms, status_str)
        where status_str is one of: "SUCCESS", "TIMEOUT", "REFUSED", "FAILED"
    """
    clean_host = "127.0.0.1" if host in ("0.0.0.0", "") else host

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout_seconds)
    t0 = time.perf_counter()
    try:
        s.connect((clean_host, port))
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return round(latency_ms, 2), "SUCCESS"
    except socket.timeout:
        return None, "TIMEOUT"
    except ConnectionRefusedError:
        return None, "REFUSED"
    except OSError as e:
        return None, f"FAILED ({e.strerror or str(e)})"
    except Exception as e:
        return None, f"FAILED ({str(e)})"
    finally:
        try:
            s.close()
        except Exception:
            pass


def measure_ping(
    host: str,
    port: Optional[int] = None,
    timeout_seconds: float = 1.0,
) -> Dict[str, Any]:
    """Perform comprehensive network ping check (both ICMP and TCP if port given).

    Returns a rich dict with ping_ms, icmp_ms, tcp_ms, status, and timestamp.
    """
    icmp_ms = measure_icmp_ping(host, timeout_seconds=timeout_seconds)
    tcp_ms = None
    tcp_status = "N/A"

    if port is not None and port > 0:
        tcp_ms, tcp_status = measure_tcp_ping(host, port, timeout_seconds=timeout_seconds)

    # Determine primary ping_ms and overall status
    if icmp_ms is not None:
        primary_ping = icmp_ms
        overall_status = "SUCCESS"
    elif tcp_ms is not None:
        primary_ping = tcp_ms
        overall_status = "SUCCESS"
    else:
        primary_ping = None
        if tcp_status in ("TIMEOUT", "REFUSED"):
            overall_status = tcp_status
        else:
            overall_status = "UNREACHABLE"

    return {
        "host": host,
        "port": port,
        "ping_ms": round(primary_ping, 2) if primary_ping is not None else None,
        "icmp_ms": round(icmp_ms, 2) if icmp_ms is not None else None,
        "tcp_ms": round(tcp_ms, 2) if tcp_ms is not None else None,
        "status": overall_status,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
    }


class ClientPingLogger:
    """Dedicated ping logger for a federated learning client.

    Maintains:
    1. Human-readable text log: client_<id>_ping.log
    2. Structured CSV log: client_<id>_ping_metrics.csv
    3. In-memory history for programmatic export and summary analysis.
    """

    def __init__(
        self,
        client_id: int,
        server_address: str,
        device_type: str = "pc",
        log_dir: Optional[Union[str, Path]] = None,
        enabled: bool = True,
    ):
        self.client_id = client_id
        self.server_address = server_address
        self.host, self.port = parse_server_address(server_address)
        self.device_type = device_type
        self.enabled = enabled

        self.log_dir = Path(log_dir) if log_dir else (Path("results") / "clients" / f"client_{client_id}")
        self.log_file = self.log_dir / f"client_{client_id}_ping.log"
        self.csv_file = self.log_dir / f"client_{client_id}_ping_metrics.csv"

        self.history: List[Dict[str, Any]] = []

        if self.enabled:
            self._ensure_files()

    def _ensure_files(self):
        """Ensure parent directory and CSV header exist."""
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            if not self.csv_file.exists():
                with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp",
                        "round",
                        "client_id",
                        "device_type",
                        "event",
                        "server_host",
                        "server_port",
                        "ping_ms",
                        "icmp_ms",
                        "tcp_ms",
                        "status",
                    ])
        except Exception as e:
            print(f"[Client {self.client_id}] Warning: Failed to initialize ping log files: {e}")

    def log_ping(
        self,
        server_round: Optional[int] = None,
        event: str = "fit",
        timeout_seconds: float = 1.0,
    ) -> Dict[str, Any]:
        """Measure latency to FL server and record to text log and CSV.

        Args:
            server_round: FL round number (e.g. 1, 2... or 0 for initialization)
            event: Event type ('init', 'fit', 'evaluate', 'heartbeat')
            timeout_seconds: Timeout for ping probes

        Returns:
            Dict containing ping measurements and status.
        """
        ping_res = measure_ping(
            host=self.host,
            port=self.port,
            timeout_seconds=timeout_seconds,
        )

        r_num = server_round if server_round is not None else 0
        record = {
            "timestamp": ping_res["timestamp"],
            "round": r_num,
            "client_id": self.client_id,
            "device_type": self.device_type,
            "event": event,
            "server_host": self.host,
            "server_port": self.port,
            "ping_ms": ping_res["ping_ms"],
            "icmp_ms": ping_res["icmp_ms"],
            "tcp_ms": ping_res["tcp_ms"],
            "status": ping_res["status"],
        }
        self.history.append(record)

        # Console message
        p_display = f"{ping_res['ping_ms']:.2f} ms" if ping_res["ping_ms"] is not None else "TIMEOUT/UNREACHABLE"
        r_label = f"Round {r_num}" if r_num > 0 else "Init"
        print(
            f"[Client {self.client_id}] [Ping] {r_label} ({event}) -> "
            f"{self.host}:{self.port} | Latency: {p_display} | Status: {ping_res['status']}"
        )

        if not self.enabled:
            return record

        # Append to human-readable log file
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                icmp_str = f"{ping_res['icmp_ms']}ms" if ping_res["icmp_ms"] is not None else "N/A"
                tcp_str = f"{ping_res['tcp_ms']}ms" if ping_res["tcp_ms"] is not None else "N/A"
                f.write(
                    f"[{ping_res['timestamp']}] [Client {self.client_id}] [{r_label}] [{event.upper()}] "
                    f"Target: {self.host}:{self.port} | Ping: {p_display} | "
                    f"Status: {ping_res['status']} (ICMP: {icmp_str}, TCP: {tcp_str})\n"
                )
        except Exception as e:
            print(f"[Client {self.client_id}] Warning: could not write to ping.log: {e}")

        # Append to CSV file
        try:
            with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    record["timestamp"],
                    record["round"],
                    record["client_id"],
                    record["device_type"],
                    record["event"],
                    record["server_host"],
                    record["server_port"],
                    record["ping_ms"] if record["ping_ms"] is not None else "",
                    record["icmp_ms"] if record["icmp_ms"] is not None else "",
                    record["tcp_ms"] if record["tcp_ms"] is not None else "",
                    record["status"],
                ])
        except Exception as e:
            print(f"[Client {self.client_id}] Warning: could not append to ping CSV: {e}")

        return record

    def get_summary(self) -> Dict[str, Any]:
        """Compute ping summary statistics across all logged events."""
        valid_pings = [r["ping_ms"] for r in self.history if r.get("ping_ms") is not None]
        if not valid_pings:
            return {
                "client_id": self.client_id,
                "total_probes": len(self.history),
                "success_count": 0,
                "packet_loss_percent": 100.0 if self.history else 0.0,
                "avg_ping_ms": None,
                "min_ping_ms": None,
                "max_ping_ms": None,
            }

        import numpy as np
        return {
            "client_id": self.client_id,
            "total_probes": len(self.history),
            "success_count": len(valid_pings),
            "packet_loss_percent": round(100.0 * (1.0 - len(valid_pings) / len(self.history)), 2),
            "avg_ping_ms": round(float(np.mean(valid_pings)), 2),
            "min_ping_ms": round(float(np.min(valid_pings)), 2),
            "max_ping_ms": round(float(np.max(valid_pings)), 2),
        }
