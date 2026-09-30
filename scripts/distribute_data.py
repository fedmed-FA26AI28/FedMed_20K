"""Automated data partition distribution from host to 10 Jetson devices via Wi-Fi/LAN (SCP).

Cluster Topology:
- 5 NVIDIA Jetson Orin (Client 0 -> 4)
- 5 NVIDIA Jetson Nano (Client 5 -> 9)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Ensure Windows console encoding does not fail on non-ASCII characters
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ==============================================================================
# 10-Device Jetson Physical Cluster Registry (5 Orin + 5 Nano)
# Note: Ensure IP addresses, usernames, and remote paths match your edge cluster.
# ==============================================================================
JETSON_CLIENTS = [
    # --- 5 NVIDIA Jetson Orin Devices (Client 0 -> 4) ---
    {
        "client_id": 0,
        "device_type": "Jetson Orin",
        "ip": "192.168.1.50",
        "username": "jetson",
        "remote_dir": "~/FedMedAI/data/"
    },
    {
        "client_id": 1,
        "device_type": "Jetson Orin",
        "ip": "192.168.1.51",
        "username": "jetson",
        "remote_dir": "~/FedMedAI/data/"
    },
    {
        "client_id": 2,
        "device_type": "Jetson Orin",
        "ip": "192.168.1.52",
        "username": "jetson",
        "remote_dir": "~/FedMedAI/data/"
    },
    {
        "client_id": 3,
        "device_type": "Jetson Orin",
        "ip": "192.168.1.53",
        "username": "jetson",
        "remote_dir": "~/FedMedAI/data/"
    },
    {
        "client_id": 4,
        "device_type": "Jetson Orin",
        "ip": "192.168.1.54",
        "username": "jetson",
        "remote_dir": "~/FedMedAI/data/"
    },

    # --- 5 NVIDIA Jetson Nano Devices (Client 5 -> 9) ---
    {
        "client_id": 5,
        "device_type": "Jetson Nano",
        "ip": "192.168.1.60",
        "username": "nano",
        "remote_dir": "~/FedMedAI/data/"
    },
    {
        "client_id": 6,
        "device_type": "Jetson Nano",
        "ip": "192.168.1.61",
        "username": "nano",
        "remote_dir": "~/FedMedAI/data/"
    },
    {
        "client_id": 7,
        "device_type": "Jetson Nano",
        "ip": "192.168.1.62",
        "username": "nano",
        "remote_dir": "~/FedMedAI/data/"
    },
    {
        "client_id": 8,
        "device_type": "Jetson Nano",
        "ip": "192.168.1.63",
        "username": "nano",
        "remote_dir": "~/FedMedAI/data/"
    },
    {
        "client_id": 9,
        "device_type": "Jetson Nano",
        "ip": "192.168.1.64",
        "username": "nano",
        "remote_dir": "~/FedMedAI/data/"
    },
]


def check_ping(ip: str) -> bool:
    """Check whether edge device is powered on and reachable on the network.

    Returns True if ping response is received, False otherwise.
    """
    param = "-n 1" if sys.platform.startswith("win") else "-c 1"
    timeout_param = "-w 1000" if sys.platform.startswith("win") else "-W 1"
    command = f"ping {param} {timeout_param} {ip}"
    result = subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def push_data_to_client(client_info: dict, local_data_root: Path) -> bool:
    """Transfer client partitioned dataset to the corresponding remote Jetson device.

    Workflow:
    1. Verify local client data folder exists.
    2. Ping target IP for network reachability.
    3. Ensure destination directory exists on target device via SSH.
    4. Secure copy (SCP) the data folder to target device.
    """
    cid = client_info["client_id"]
    ip = client_info["ip"]
    user = client_info["username"]
    dev_type = client_info["device_type"]
    remote_dir = client_info["remote_dir"]

    local_client_data = local_data_root / f"client_{cid}"

    print("-" * 65)
    print(f"Client {cid} | Type: {dev_type} | Target: {user}@{ip}")
    print("-" * 65)

    # Step 1: Check local client partition folder
    if not local_client_data.exists():
        print(f"ERROR: Local data directory not found: {local_client_data}")
        print("Please run data partitioning script (datasets/partition.py) first.")
        return False

    # Step 2: Test ping reachability
    print(f"Checking network connection to {ip}...")
    if not check_ping(ip):
        print(f"WARNING: Cannot ping {ip}. The device might be offline or IP is incorrect.")
        choice = input("Do you want to proceed with SCP anyway? (y/n): ").strip().lower()
        if choice != "y":
            print("Skipped this device.")
            return False

    # Step 3: Create target directory on remote host
    ssh_mkdir_cmd = f'ssh -o StrictHostKeyChecking=no {user}@{ip} "mkdir -p {remote_dir}"'
    subprocess.run(ssh_mkdir_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Step 4: Transfer data via SCP
    remote_target = f"{user}@{ip}:{remote_dir}"
    scp_cmd = f"scp -r {local_client_data} {remote_target}"

    print(f"Transferring {local_client_data.name} to {remote_target}...")
    ret = subprocess.run(scp_cmd, shell=True)

    if ret.returncode == 0:
        print(f"Successfully transferred data for Client {cid} ({dev_type}).")
        return True
    else:
        print(f"Error transferring data to {ip} (Exit code: {ret.returncode}).")
        return False


def main():
    """Main CLI driver for network data distribution:
    - Parses command line parameters (--client_id, --data_dir).
    - Iterates over Jetson device targets and initiates secure copy.
    - Aggregates final deployment status.
    """
    parser = argparse.ArgumentParser(description="Distribute FL partitioned datasets to 10 Jetson devices over Wi-Fi.")
    parser.add_argument(
        "--client_id", 
        type=int, 
        default=None, 
        help="Target client ID (0 to 9). Default is all 10 devices."
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data", 
        help="Path to partitioned data directory on host (default: 'data')."
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    local_data_root = project_root / args.data_dir

    print("=" * 65)
    print("             FEDMEDAI DATA DISTRIBUTION OVER NETWORK             ")
    print(f"             Total devices: {len(JETSON_CLIENTS)} (5 Orin + 5 Nano)              ")
    print("=" * 65)

    if args.client_id is not None:
        target_clients = [c for c in JETSON_CLIENTS if c["client_id"] == args.client_id]
        if not target_clients:
            print(f"Invalid Client ID: {args.client_id}. Must be between 0 and {len(JETSON_CLIENTS)-1}.")
            return
    else:
        target_clients = JETSON_CLIENTS

    success_count = 0
    total_count = len(target_clients)

    for client in target_clients:
        if push_data_to_client(client, local_data_root):
            success_count += 1

    print("=" * 65)
    print(f"RESULT: Completed {success_count}/{total_count} devices successfully.")
    print("=" * 65)


if __name__ == "__main__":
    main()
