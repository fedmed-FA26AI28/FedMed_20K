"""Evaluate validation-selected checkpoints once on the official test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.data import get_medmnist_datasets
from research.fast_data import FastImageDataset
from research.run import ClientShiftDataset, evaluate, evaluation_transform, validation_groups
from research.select import file_sha256
from research.support_calibrated import make_model


def write_json_atomically(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    lock = json.loads(args.lock_file.read_text(encoding="utf-8"))
    lock_digest = file_sha256(args.lock_file)
    output = args.lock_file.with_name(args.lock_file.stem + "_test.json")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    dataset_cache = {}
    completed = (json.loads(output.read_text(encoding="utf-8")) if output.exists() else
                 {"lock_file": str(args.lock_file), "selection_metric": lock["selection_metric"],
                  "lock_sha256": lock_digest,
                  "methods": {}, "summary": {}, "test_evaluations": 0})
    if completed["lock_file"] != str(args.lock_file):
        raise ValueError("existing test result belongs to a different lock file")
    if completed.get("lock_sha256") not in (None, lock_digest):
        raise ValueError("validation lock changed after test evaluation began")
    for method, selection in lock["methods"].items():
        rows = completed["methods"].get(method, [])
        for run in selection["runs"]:
            if any(row["seed"] == run["seed"] for row in rows):
                continue
            for name in ("development", "checkpoint"):
                expected = run.get(name + "_sha256")
                if expected and file_sha256(Path(run[name])) != expected:
                    raise ValueError(f"locked {name} changed after validation selection: {run[name]}")
            development = json.loads(Path(run["development"]).read_text(encoding="utf-8"))
            cfg = development["configuration"]
            key = (cfg["dataset"], cfg["resolution"], cfg.get("fast_data", False))
            if key not in dataset_cache:
                # Train and validation are never loaded by this evaluator.
                _, _, test, classes = get_medmnist_datasets(
                    cfg["dataset"], download=False, augment=False,
                    load_train=False, load_val=False, load_test=True,
                )
                if cfg.get("fast_data", False):
                    test = FastImageDataset(test, cfg["resolution"])
                else:
                    test.transform = evaluation_transform(test, cfg["resolution"])
                dataset_cache[key] = test, classes
            test, classes = dataset_cache[key]
            client_groups = None
            if cfg.get("client_shift_strength", 0):
                client_groups = validation_groups(
                    test, np.asarray(development["train_class_counts"]), cfg["seed"])
                test = ClientShiftDataset(test, client_groups, cfg["seed"],
                                          cfg["client_shift_strength"])
            model = make_model(classes, cfg.get("norm", "gn")).to(device)
            saved = torch.load(run["checkpoint"], map_location="cpu", weights_only=True)
            model.load_state_dict(saved["state_dict"])
            score = evaluate(model, test, device, cfg["batch_size"], client_groups)
            completed["test_evaluations"] += 1
            rows.append({"seed": run["seed"], "selected_round": saved["round"], "metrics": score})
            # Persist every completed evaluation so interrupted finalization
            # resumes without evaluating the same checkpoint again.
            completed["methods"][method] = rows
            write_json_atomically(output, completed)
            print(f"{method} seed {run['seed']}: test macro-F1={score['macro_f1']:.4f}", flush=True)
        completed["summary"][method] = {
            "macro_f1_mean": float(np.mean([row["metrics"]["macro_f1"] for row in rows])),
            "macro_f1_sd": float(np.std([row["metrics"]["macro_f1"] for row in rows], ddof=1))
            if len(rows) > 1 else 0.0,
            "accuracy_mean": float(np.mean([row["metrics"]["accuracy"] for row in rows])),
        }
    write_json_atomically(output, completed)
    print(f"Final test summary: {output}")


if __name__ == "__main__":
    main()
