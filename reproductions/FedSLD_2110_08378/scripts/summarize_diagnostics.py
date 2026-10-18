"""Print a compact comparison of FedSLD reconstruction result JSON files."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.results_root.rglob("*.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if result.get("study") != "FedSLD independent reconstruction":
            continue
        history = result.get("history") or []
        if not history:
            continue
        losses = [float(entry["train_loss"]) for entry in history]
        rows.append(
            {
                "split": result["dataset"].get("split_mode", "official"),
                "algorithm": result["config"]["algorithm"],
                "rounds": result["config"]["rounds"],
                "bta": 100 * float(result["best_test_accuracy"]),
                "best_round": result["best_test_accuracy_round"],
                "final": 100 * float(history[-1]["combined_test"]["accuracy"]),
                "max_loss": max(losses),
                "hypothesis": bool(result.get("reconstruction_hypothesis", False)),
                "path": path,
            }
        )

    if not rows:
        raise SystemExit(f"No reconstruction result JSON files found under {args.results_root}")

    header = (
        f"{'split':<11} {'algorithm':<22} {'rounds':>6} {'BTA%':>8} "
        f"{'best':>6} {'final%':>8} {'max_loss':>10} {'hypothesis':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda value: (value["split"], value["algorithm"])):
        print(
            f"{row['split']:<11} {row['algorithm']:<22} {row['rounds']:>6} "
            f"{row['bta']:>8.2f} {row['best_round']:>6} {row['final']:>8.2f} "
            f"{row['max_loss']:>10.4f} {str(row['hypothesis']):>11}"
        )
        print(f"  {row['path']}")


if __name__ == "__main__":
    main()
