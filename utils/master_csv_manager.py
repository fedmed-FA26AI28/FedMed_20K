"""Master CSV Manager for FedMedAI Experiments.

Maintains and updates the master benchmark tracking CSV ('CSV file/master_runs.csv')
across multiple runs (run_1 to run_5) for both Centralized baselines and Federated Learning strategies.
Calculates sample-weighted evaluation loss (Loss), BTA, BMCTA, runtime,
and automatically computes the mean/average row for each 5-run group separated by a blank line.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Canonical Columns matching the project template with single sample-weighted Loss
MASTER_COLUMNS = [
    "dataset",
    "partition_method",
    "strategy",
    "model",
    "run",
    "num_params",
    "model_size_mb",
    "total_comm_mb",
    "total_time_min",
    "total_time_s",
    "BTA",
    "BMCTA",
    "Loss",
    "optimizer",
    "learning_rate",
]

# Dataset configurations (num_classes and in_channels for MedMNIST)
DATASET_SPECS = {
    "bloodmnist": {"num_classes": 8, "in_channels": 3},
    "pathmnist": {"num_classes": 9, "in_channels": 3},
    "organamnist": {"num_classes": 11, "in_channels": 3},  # Repeated to 3 channels in preprocessing
    "dermamnist": {"num_classes": 7, "in_channels": 3},
}

# Pre-computed parameter counts and float32 sizes
MODEL_SPECS_CACHE = {
    ("baseline", "bloodmnist"): (570204, 2.1752),
    ("baseline", "pathmnist"): (570705, 2.1771),
    ("baseline", "organamnist"): (571707, 2.1809),
    ("baseline", "dermamnist"): (569703, 2.1732),
    ("baseline_bn", "bloodmnist"): (570396, 2.1759),
    ("baseline_bn", "pathmnist"): (570897, 2.1778),
    ("baseline_bn", "organamnist"): (571899, 2.1816),
    ("baseline_bn", "dermamnist"): (569895, 2.1740),
    ("cnn", "bloodmnist"): (20104, 0.0767),
    ("cnn", "pathmnist"): (20169, 0.0769),
    ("cnn", "organamnist"): (20299, 0.0774),
    ("cnn", "dermamnist"): (20039, 0.0764),
}


def normalize_strategy_name(strategy: str) -> str:
    s = str(strategy).lower().strip()
    if s in ("fedbns", "fedbn_scaffold", "proposed"):
        return "proposed"
    if s in ("centralized", "cen", "central"):
        return "centralized"
    return s


def normalize_partition_name(partition: str) -> str:
    p = str(partition).lower().strip()
    if p in ("centralized", "cen", "none"):
        return "centralized"
    return p


def normalize_model_name(model: str) -> str:
    m = str(model).lower().strip()
    if m in ("baseline_bn", "baselinecnn_bn", "classic_bn"):
        return "baseline_bn"
    if m in ("baseline", "paper", "classic_cnn", "classic"):
        return "baseline"
    if m in ("cnn", "simple_cnn", "modern_cnn"):
        return "cnn"
    return m


def normalize_run_id(run_id: str) -> str:
    r = str(run_id).strip().lower()
    if r in ("mean", "average", "avg"):
        return "average"
    
    # Check if already in format run_X_seed_Y
    m_full = re.match(r"^run_?(\d+)_seed_?(\d+)$", r)
    if m_full:
        return f"run_{m_full.group(1)}_seed_{m_full.group(2)}"

    # Match simple number or run_X / runX
    m_simple = re.match(r"^(?:run_?)?(\d+)$", r)
    if m_simple:
        num = int(m_simple.group(1))
        seed = 42 + (num - 1)
        return f"run_{num}_seed_{seed}"

    return r


def get_model_stats(model: str, dataset: str) -> Tuple[int, float]:
    m = normalize_model_name(model)
    d = dataset.lower().strip()
    if (m, d) in MODEL_SPECS_CACHE:
        return MODEL_SPECS_CACHE[(m, d)]
    
    try:
        from models.factory import get_model
        cfg = DATASET_SPECS.get(d, {"num_classes": 8, "in_channels": 3})
        instance = get_model(m, num_classes=cfg["num_classes"], in_channels=cfg["in_channels"])
        params = sum(p.numel() for p in instance.parameters())
        size_mb = sum(p.numel() * p.element_size() for p in instance.parameters()) / (1024 * 1024)
        return params, round(size_mb, 4)
    except Exception:
        return 570705, 2.18


def generate_initial_template_rows() -> List[Dict[str, Any]]:
    """Builds pre-populated benchmark template rows including Centralized and Federated strategies."""
    datasets = ["bloodmnist", "pathmnist", "organamnist"]
    models = ["baseline", "baseline_bn"]
    runs = [
        "run_1_seed_42",
        "run_2_seed_43",
        "run_3_seed_44",
        "run_4_seed_45",
        "run_5_seed_46",
        "average",
    ]
    rows = []

    for d in datasets:
        # 1. Centralized Upper-Bound Baselines
        for m in models:
            num_params, model_size_mb = get_model_stats(m, d)
            for r in runs:
                rows.append({
                    "dataset": d,
                    "partition_method": "centralized",
                    "strategy": "centralized",
                    "model": m,
                    "run": r,
                    "num_params": num_params,
                    "model_size_mb": model_size_mb,
                    "total_comm_mb": "0.00",
                    "total_time_min": "",
                    "total_time_s": "",
                    "BTA": "",
                    "BMCTA": "",
                    "Loss": "",
                    "optimizer": "sgd",
                    "learning_rate": 0.01,
                })

        # 2. Federated Learning Strategies (practical partition)
        fl_strategies = ["fedavg", "fedprox", "fedbn", "scaffold", "proposed"]
        for s in fl_strategies:
            for m in models:
                num_params, model_size_mb = get_model_stats(m, d)
                for r in runs:
                    rows.append({
                        "dataset": d,
                        "partition_method": "practical",
                        "strategy": s,
                        "model": m,
                        "run": r,
                        "num_params": num_params,
                        "model_size_mb": model_size_mb,
                        "total_comm_mb": "",
                        "total_time_min": "",
                        "total_time_s": "",
                        "BTA": "",
                        "BMCTA": "",
                        "Loss": "",
                        "optimizer": "sgd",
                        "learning_rate": 0.01,
                    })

        # 3. Dirichlet Partitions (dirichlet_0.1 and dirichlet_0.3)
        # Exclude bloodmnist until target hardware (e.g. Jetson Orin) client count is determined
        if d != "bloodmnist":
            dirichlet_partitions = ["dirichlet_0.1", "dirichlet_0.3"]
            for p in dirichlet_partitions:
                for s in fl_strategies:
                    for m in models:
                        num_params, model_size_mb = get_model_stats(m, d)
                        for r in runs:
                            rows.append({
                                "dataset": d,
                                "partition_method": p,
                                "strategy": s,
                                "model": m,
                                "run": r,
                                "num_params": num_params,
                                "model_size_mb": model_size_mb,
                                "total_comm_mb": "",
                                "total_time_min": "",
                                "total_time_s": "",
                                "BTA": "",
                                "BMCTA": "",
                                "Loss": "",
                                "optimizer": "sgd",
                                "learning_rate": 0.01,
                            })

    return rows


def save_master_csv_with_spacing(df: pd.DataFrame, csv_path: Path):
    """Saves DataFrame to CSV with a blank row separating distinct experiment groups (5 runs + average)."""
    csv_path = Path(csv_path)
    if csv_path.suffix == ".xlsx":
        export_master_excel(df, csv_path)
        return
    lines = [",".join(MASTER_COLUMNS)]

    # Discover unique groups preserving logical order
    groups = []
    seen = set()
    for _, row in df.iterrows():
        d_val = str(row.get("dataset", "")).lower().strip()
        p_val = str(row.get("partition_method", "")).lower().strip()
        s_val = normalize_strategy_name(str(row.get("strategy", "")))
        m_val = normalize_model_name(str(row.get("model", "")))
        if not d_val:
            continue
        key = (d_val, p_val, s_val, m_val)
        if key not in seen:
            seen.add(key)
            groups.append(key)

    for key in groups:
        d, p, s, m = key
        mask = (
            (df["dataset"].str.lower().str.strip() == d) &
            (df["partition_method"].str.lower().str.strip() == p) &
            (df["strategy"].apply(normalize_strategy_name) == s) &
            (df["model"].apply(normalize_model_name) == m)
        )
        group_df = df[mask]
        run_order = [
            "run_1_seed_42",
            "run_2_seed_43",
            "run_3_seed_44",
            "run_4_seed_45",
            "run_5_seed_46",
            "average",
        ]

        ordered_rows = []
        for r in run_order:
            r_match = group_df[group_df["run"].apply(normalize_run_id) == r]
            if not r_match.empty:
                ordered_rows.append(r_match.iloc[0])

        other_matches = group_df[~group_df["run"].apply(normalize_run_id).isin(run_order)]
        for _, r_row in other_matches.iterrows():
            ordered_rows.append(r_row)

        for row in ordered_rows:
            vals = [str(row.get(col, "")).replace(",", " ") for col in MASTER_COLUMNS]
            lines.append(",".join(vals))

        # Blank line between 5-run+average blocks
        lines.append("")

    content = "\n".join(lines)
    temp_csv = csv_path.with_suffix(".tmp")
    with open(temp_csv, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    temp_csv.replace(csv_path)

    # Automatically generate / update master_runs.xlsx with Sheet 1 ('All Runs') and Sheet 2 ('Average Only')
    try:
        export_master_excel(df, csv_path.parent / "master_runs.xlsx")
    except Exception as e:
        print(f"  [Notice] Could not update Excel workbook: {e}")


def export_master_excel(df: pd.DataFrame, excel_path: Path):
    """Exports 'All Runs', 'Average Summary', 'Dirichlet Alpha', and 'Dirichlet Summary' sheets into a cleanly formatted Excel workbook."""
    excel_path = Path(excel_path)
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    run_order = [
        "run_1_seed_42",
        "run_2_seed_43",
        "run_3_seed_44",
        "run_4_seed_45",
        "run_5_seed_46",
        "average",
    ]

    # Discover unique groups preserving logical order
    groups = []
    seen = set()
    for _, row in df.iterrows():
        d_val = str(row.get("dataset", "")).lower().strip()
        p_val = str(row.get("partition_method", "")).lower().strip()
        s_val = normalize_strategy_name(str(row.get("strategy", "")))
        m_val = normalize_model_name(str(row.get("model", "")))
        if not d_val:
            continue
        key = (d_val, p_val, s_val, m_val)
        if key not in seen:
            seen.add(key)
            groups.append(key)

    # 1. Sheet 1: All Runs (Centralized & Practical)
    groups_practical = [g for g in groups if g[1] in ("centralized", "practical")]
    sheet1_rows = []
    for key in groups_practical:
        d, p, s, m = key
        mask = (
            (df["dataset"].str.lower().str.strip() == d) &
            (df["partition_method"].str.lower().str.strip() == p) &
            (df["strategy"].apply(normalize_strategy_name) == s) &
            (df["model"].apply(normalize_model_name) == m)
        )
        group_df = df[mask]
        for r in run_order:
            r_match = group_df[group_df["run"].apply(normalize_run_id) == r]
            if not r_match.empty:
                sheet1_rows.append(r_match.iloc[0].to_dict())
        sheet1_rows.append({col: "" for col in MASTER_COLUMNS})
    sheet1_df = pd.DataFrame(sheet1_rows, columns=MASTER_COLUMNS)

    # 2. Sheet 2: Average Summary (Centralized & Practical)
    avg_mask = (df["run"].apply(normalize_run_id) == "average") & (df["partition_method"].apply(normalize_partition_name).isin(["centralized", "practical"]))
    raw_avg_df = df[avg_mask].copy()
    sheet2_rows = []
    current_d = None
    for _, row in raw_avg_df.iterrows():
        d_val = str(row.get("dataset", "")).lower().strip()
        if current_d is not None and d_val != current_d:
            sheet2_rows.append({col: "" for col in MASTER_COLUMNS})
        current_d = d_val
        sheet2_rows.append(row.to_dict())
    sheet2_df = pd.DataFrame(sheet2_rows, columns=MASTER_COLUMNS)

    # 3. Sheet 3: Dirichlet Alpha (All Runs for dirichlet_0.1 and dirichlet_0.3)
    groups_dirichlet = [g for g in groups if str(g[1]).startswith("dirichlet")]
    sheet3_rows = []
    for key in groups_dirichlet:
        d, p, s, m = key
        mask = (
            (df["dataset"].str.lower().str.strip() == d) &
            (df["partition_method"].str.lower().str.strip() == p) &
            (df["strategy"].apply(normalize_strategy_name) == s) &
            (df["model"].apply(normalize_model_name) == m)
        )
        group_df = df[mask]
        for r in run_order:
            r_match = group_df[group_df["run"].apply(normalize_run_id) == r]
            if not r_match.empty:
                sheet3_rows.append(r_match.iloc[0].to_dict())
        sheet3_rows.append({col: "" for col in MASTER_COLUMNS})
    sheet3_df = pd.DataFrame(sheet3_rows, columns=MASTER_COLUMNS)

    # 4. Sheet 4: Dirichlet Summary (Only Averages for dirichlet_0.1 and dirichlet_0.3)
    avg_dir_mask = (df["run"].apply(normalize_run_id) == "average") & (df["partition_method"].str.startswith("dirichlet"))
    raw_avg_dir_df = df[avg_dir_mask].copy()
    sheet4_rows = []
    current_dp = None
    for _, row in raw_avg_dir_df.iterrows():
        d_val = str(row.get("dataset", "")).lower().strip()
        p_val = str(row.get("partition_method", "")).lower().strip()
        dp_key = (d_val, p_val)
        if current_dp is not None and dp_key != current_dp:
            sheet4_rows.append({col: "" for col in MASTER_COLUMNS})
        current_dp = dp_key
        sheet4_rows.append(row.to_dict())
    sheet4_df = pd.DataFrame(sheet4_rows, columns=MASTER_COLUMNS)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        sheet1_df.to_excel(writer, sheet_name="All Runs", index=False)
        sheet2_df.to_excel(writer, sheet_name="Average Summary", index=False)
        sheet3_df.to_excel(writer, sheet_name="Dirichlet Alpha", index=False)
        sheet4_df.to_excel(writer, sheet_name="Dirichlet Summary", index=False)

    # Style workbook with professional borders and highlights
    try:
        import openpyxl
        from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
        from openpyxl.utils import get_column_letter

        wb = openpyxl.load_workbook(excel_path)

        header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")

        avg_fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
        avg_font = Font(name="Calibri", size=10, bold=True, color="276A3C")

        normal_font = Font(name="Calibri", size=10, color="000000")
        thin_side = Side(border_style="thin", color="D9D9D9")
        thick_bottom_side = Side(border_style="medium", color="1F497D")

        thin_border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)
        header_border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thick_bottom_side)

        center_align = Alignment(horizontal="center", vertical="center")
        left_align = Alignment(horizontal="left", vertical="center")
        right_align = Alignment(horizontal="right", vertical="center")

        num_cols = ["num_params", "model_size_mb", "total_comm_mb", "total_time_min", "total_time_s", "BTA", "BMCTA", "Loss", "learning_rate"]
        text_left_cols = ["dataset", "partition_method", "strategy", "model", "optimizer"]

        for sheetname in wb.sheetnames:
            ws = wb[sheetname]
            ws.views.sheetView[0].showGridLines = True

            # Format headers
            for col_idx in range(1, ws.max_column + 1):
                cell = ws.cell(row=1, column=col_idx)
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = center_align
                cell.border = header_border

            # Format data rows
            for row_idx in range(2, ws.max_row + 1):
                d_val = str(ws.cell(row=row_idx, column=1).value or "").strip()
                r_val = str(ws.cell(row=row_idx, column=5).value or "").strip().lower()

                # If this is a blank separator line
                if not d_val and not r_val:
                    ws.row_dimensions[row_idx].height = 12
                    for col_idx in range(1, ws.max_column + 1):
                        cell = ws.cell(row=row_idx, column=col_idx)
                        cell.fill = PatternFill(fill_type=None)
                        cell.border = Border()
                    continue

                is_avg = ("average" in r_val)

                for col_idx in range(1, ws.max_column + 1):
                    cell = ws.cell(row=row_idx, column=col_idx)
                    col_header = str(ws.cell(row=1, column=col_idx).value or "")

                    if is_avg:
                        cell.fill = avg_fill
                        cell.font = avg_font
                        cell.border = thin_border
                    else:
                        cell.fill = PatternFill(fill_type=None)
                        cell.font = normal_font
                        cell.border = thin_border

                    if col_header in num_cols:
                        cell.alignment = right_align
                    elif col_header in text_left_cols:
                        cell.alignment = left_align
                    else:
                        cell.alignment = center_align

            # Auto-fit column widths
            for col in ws.columns:
                max_len = 0
                col_letter = get_column_letter(col[0].column)
                for cell in col:
                    val_str = str(cell.value or "")
                    if len(val_str) > max_len:
                        max_len = len(val_str)
                ws.column_dimensions[col_letter].width = max(max_len + 4, 12)

        wb.save(excel_path)
    except Exception as err:
        print(f"  [Notice] Openpyxl styling skipped: {err}")


def load_master_dataframe(csv_path: Path) -> Tuple[pd.DataFrame, Path]:
    """Loads current master DataFrame from master_runs.xlsx or CSV, ensuring missing template rows are appended."""
    csv_path = Path(csv_path)
    excel_path = csv_path.parent / "master_runs.xlsx" if csv_path.suffix == ".csv" else csv_path

    df = None
    if excel_path.exists():
        try:
            xl = pd.ExcelFile(excel_path)
            dfs = []
            for sname in ["All Runs", "Dirichlet Alpha"]:
                if sname in xl.sheet_names:
                    sdf = pd.read_excel(excel_path, sheet_name=sname, dtype=str).fillna("")
                    sdf = sdf[sdf["dataset"].astype(str).str.strip() != ""]
                    dfs.append(sdf)
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
            elif xl.sheet_names:
                df = pd.read_excel(excel_path, sheet_name=0, dtype=str).fillna("")
                df = df[df["dataset"].astype(str).str.strip() != ""]
        except Exception as e:
            print(f"  [Notice] Could not load from {excel_path}: {e}")
            df = None

    if df is None or df.empty:
        if csv_path.exists() and csv_path.stat().st_size > 0:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False).dropna(how="all")
            df = df[df["dataset"].astype(str).str.strip() != ""]
        else:
            rows = generate_initial_template_rows()
            df = pd.DataFrame(rows, columns=MASTER_COLUMNS)

    # Ensure all canonical template rows (including new Dirichlet rows) exist
    existing_keys = set(zip(
        df["dataset"].str.lower().str.strip(),
        df["partition_method"].apply(normalize_partition_name),
        df["strategy"].apply(normalize_strategy_name),
        df["model"].apply(normalize_model_name),
        df["run"].apply(normalize_run_id)
    ))
    missing_rows = []
    for trow in generate_initial_template_rows():
        tkey = (
            trow["dataset"].lower().strip(),
            normalize_partition_name(trow["partition_method"]),
            normalize_strategy_name(trow["strategy"]),
            normalize_model_name(trow["model"]),
            normalize_run_id(trow["run"])
        )
        if tkey not in existing_keys:
            missing_rows.append(trow)
    if missing_rows:
        df = pd.concat([df, pd.DataFrame(missing_rows, columns=MASTER_COLUMNS)], ignore_index=True)

    return df, excel_path


def init_master_csv(file_path: Path) -> Path:
    """Ensures master_runs.xlsx (and optional CSV) exists. If not, creates it with full initial template and block spacing."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    excel_path = file_path.parent / "master_runs.xlsx" if file_path.suffix == ".csv" else file_path
    if not excel_path.exists() or excel_path.stat().st_size == 0:
        rows = generate_initial_template_rows()
        df = pd.DataFrame(rows, columns=MASTER_COLUMNS)
        export_master_excel(df, excel_path)
        print(f"  [Master Tracker] Initialized template with {len(rows)} rows at: {excel_path}")
    return file_path


def update_master_csv(
    csv_path: Path,
    dataset: str,
    partition_method: str,
    strategy: str,
    model: str,
    run_id: str,
    metrics: Dict[str, Any],
    optimizer: Optional[str] = None,
    learning_rate: Optional[float] = None,
) -> Path:
    """Updates master Excel / CSV with a completed run's metrics, recalculates the average row, and formats spacing."""
    csv_path = Path(csv_path)
    init_master_csv(csv_path)

    d_norm = dataset.lower().strip()
    p_norm = normalize_partition_name(partition_method)
    s_norm = normalize_strategy_name(strategy)
    m_norm = normalize_model_name(model)
    r_norm = normalize_run_id(run_id)

    # Read from Excel workbook (or CSV fallback) ensuring existing runs are preserved and missing template rows added
    df, excel_path = load_master_dataframe(csv_path)

    for col in MASTER_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # Locate existing row for this exact combination
    mask = (
        (df["dataset"].str.lower().str.strip() == d_norm) &
        (df["partition_method"].apply(normalize_partition_name) == p_norm) &
        (df["strategy"].apply(normalize_strategy_name) == s_norm) &
        (df["model"].apply(normalize_model_name) == m_norm) &
        (df["run"].apply(normalize_run_id) == r_norm)
    )

    num_params, model_size_mb = get_model_stats(m_norm, d_norm)

    total_time_s = float(metrics.get("total_time_seconds") or metrics.get("total_wall_clock_time_seconds") or metrics.get("training_time", 0.0))
    total_time_min = round(total_time_s / 60.0, 2) if total_time_s > 0 else 0.0
    total_comm_mb = float(metrics.get("total_communication_mb", 0.0) if s_norm != "centralized" else 0.0)

    # BTA and BMCTA: Store as percentage e.g. 70.47
    bta_val = metrics.get("bta")
    if bta_val is None:
        bta_val = metrics.get("best_test_accuracy", metrics.get("accuracy"))
    if bta_val is not None:
        bta_float = float(bta_val)
        if bta_float <= 1.0:
            bta_str = f"{bta_float * 100:.2f}"
        else:
            bta_str = f"{bta_float:.2f}"
    else:
        bta_str = ""

    bmcta_val = metrics.get("bmcta")
    if bmcta_val is None and s_norm == "centralized":
        bmcta_val = bta_val  # Centralized fairness = total accuracy
    if bmcta_val is not None:
        bmcta_float = float(bmcta_val)
        if bmcta_float <= 1.0:
            bmcta_str = f"{bmcta_float * 100:.2f}"
        else:
            bmcta_str = f"{bmcta_float:.2f}"
    else:
        bmcta_str = ""

    # Sample-weighted evaluation loss: (sum(N_i * Loss_i)) / sum(N_i)
    loss_val = metrics.get("loss")
    if loss_val is None:
        loss_val = metrics.get("best_test_loss")
    if loss_val is None:
        loss_val = metrics.get("best_global_loss")
    if loss_val is None:
        loss_val = metrics.get("server_loss")
    if loss_val is None:
        loss_val = metrics.get("final_global_loss", metrics.get("final_test_loss", metrics.get("test_loss")))
    loss_str = f"{float(loss_val):.4f}" if loss_val is not None and str(loss_val) != "" else ""

    opt = str(optimizer or metrics.get("optimizer", "sgd")).lower()
    lr = str(learning_rate if learning_rate is not None else metrics.get("learning_rate", "0.01"))

    row_data = {
        "dataset": d_norm,
        "partition_method": p_norm,
        "strategy": s_norm,
        "model": m_norm,
        "run": r_norm,
        "num_params": str(num_params),
        "model_size_mb": f"{model_size_mb:.4f}",
        "total_comm_mb": f"{total_comm_mb:.2f}",
        "total_time_min": f"{total_time_min:.2f}",
        "total_time_s": f"{total_time_s:.2f}",
        "BTA": bta_str,
        "BMCTA": bmcta_str,
        "Loss": loss_str,
        "optimizer": opt,
        "learning_rate": lr,
    }

    if mask.any():
        idx = df[mask].index[0]
        for k, v in row_data.items():
            df.at[idx, k] = v
    else:
        new_row_df = pd.DataFrame([row_data], columns=MASTER_COLUMNS)
        df = pd.concat([df, new_row_df], ignore_index=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Recalculate Average / Mean Row for (dataset, partition_method, strategy, model)
    # ──────────────────────────────────────────────────────────────────────────
    run_mask = (
        (df["dataset"].str.lower().str.strip() == d_norm) &
        (df["partition_method"].apply(normalize_partition_name) == p_norm) &
        (df["strategy"].apply(normalize_strategy_name) == s_norm) &
        (df["model"].apply(normalize_model_name) == m_norm) &
        (df["run"].apply(normalize_run_id).isin([
            "run_1_seed_42",
            "run_2_seed_43",
            "run_3_seed_44",
            "run_4_seed_45",
            "run_5_seed_46",
        ]))
    )

    completed_runs = df[run_mask].copy()

    def _to_num(series):
        return pd.to_numeric(series.mask(series == "", np.nan), errors="coerce").dropna()

    bta_series = _to_num(completed_runs["BTA"])
    bmcta_series = _to_num(completed_runs["BMCTA"])
    loss_series = _to_num(completed_runs["Loss"])
    time_s_series = _to_num(completed_runs["total_time_s"])
    comm_series = _to_num(completed_runs["total_comm_mb"])

    def _format_mean_std(series: pd.Series, decimals: int = 2) -> str:
        if series.empty:
            return ""
        m = series.mean()
        if len(series) > 1:
            s = series.std(ddof=1)
            return f"{m:.{decimals}f} ± {s:.{decimals}f}"
        return f"{m:.{decimals}f}"

    avg_bta = _format_mean_std(bta_series, 2)
    avg_bmcta = _format_mean_std(bmcta_series, 2)
    avg_loss = _format_mean_std(loss_series, 4)
    avg_time_s = f"{time_s_series.mean():.2f}" if not time_s_series.empty else ""
    avg_time_min = f"{(time_s_series.mean() / 60.0):.2f}" if not time_s_series.empty else ""
    avg_comm = f"{comm_series.mean():.2f}" if not comm_series.empty else ""

    avg_mask = (
        (df["dataset"].str.lower().str.strip() == d_norm) &
        (df["partition_method"].apply(normalize_partition_name) == p_norm) &
        (df["strategy"].apply(normalize_strategy_name) == s_norm) &
        (df["model"].apply(normalize_model_name) == m_norm) &
        (df["run"].apply(normalize_run_id) == "average")
    )

    avg_data = {
        "dataset": d_norm,
        "partition_method": p_norm,
        "strategy": s_norm,
        "model": m_norm,
        "run": "average",
        "num_params": str(num_params),
        "model_size_mb": f"{model_size_mb:.4f}",
        "total_comm_mb": avg_comm,
        "total_time_min": avg_time_min,
        "total_time_s": avg_time_s,
        "BTA": avg_bta,
        "BMCTA": avg_bmcta,
        "Loss": avg_loss,
        "optimizer": opt,
        "learning_rate": lr,
    }

    if avg_mask.any():
        avg_idx = df[avg_mask].index[0]
        for k, v in avg_data.items():
            df.at[avg_idx, k] = v
    else:
        new_avg_df = pd.DataFrame([avg_data], columns=MASTER_COLUMNS)
        df = pd.concat([df, new_avg_df], ignore_index=True)

    # Save cleanly to master Excel (with 4 formatted sheets) and optional CSV
    export_master_excel(df, excel_path)
    if csv_path.suffix == ".csv":
        save_master_csv_with_spacing(df, csv_path)

    print(f"\n{'='*70}")
    print(f"  [Master CSV Updated] -> {csv_path.name}")
    print(f"  Configuration : {d_norm.upper()} | {s_norm.upper()} | {m_norm.upper()} | {p_norm}")
    print(f"  Recorded Run  : {r_norm} -> BTA={bta_str}% | BMCTA={bmcta_str}% | Loss={loss_str}")
    if avg_bta:
        print(f"  Current Mean  : average ({len(bta_series)} runs) -> BTA={avg_bta}% | BMCTA={avg_bmcta}% | Loss={avg_loss}")
    print(f"{'='*70}\n")

    return csv_path
