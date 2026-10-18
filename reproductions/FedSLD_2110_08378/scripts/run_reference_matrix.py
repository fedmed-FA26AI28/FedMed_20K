"""Print or execute the six Organ/Path FedAvg/FedProx/FedSLD jobs per partition."""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/reference")
    args = parser.parse_args()
    for dataset in ("organamnist", "pathmnist"):
        for partition in ("practical", "pathological"):
            for algorithm in ("fedavg", "fedprox", "fedsld"):
                command = [
                    sys.executable,
                    "-m",
                    "fedsld_reproduction.run",
                    "--dataset",
                    dataset,
                    "--partition",
                    partition,
                    "--algorithm",
                    algorithm,
                    "--device",
                    args.device,
                    "--seed",
                    str(args.seed),
                    "--output-dir",
                    args.output_dir,
                ]
                print(" ".join(command), flush=True)
                if args.execute:
                    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

