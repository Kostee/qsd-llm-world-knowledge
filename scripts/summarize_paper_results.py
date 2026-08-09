#!/usr/bin/env python3
"""
Summarize the final paper-oriented QSD LLM result folders.

This script keeps two concepts separate:

1. majority-vote accuracy:
   accuracy of the final `prediction` column, where each row's prediction is
   the majority vote over pred_run_1 ... pred_run_N;

2. repeat-level variability:
   mean and sample SD of the N independent prediction columns.

Only full (`limitall`) final-result families are included by default:
- balanced no-RAG
- balanced RAG
- pseudo_paper_final_model_ready no-RAG

Usage:
    python scripts/summarize_paper_results.py

Output:
    results/paper_metrics_summary.csv
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

import pandas as pd


FINAL_MODELS = [
    "GPT-5.1",
    "GPT-4o",
    "GPT-4o-mini",
    "Qwen-Max",
    "Qwen3-8B",
    "Llama 3.1 70B",
    "Llama 3.1 8B",
]


def canonical_model(raw: str) -> str:
    low = raw.lower().replace("_", "-")
    if "gpt-5.1" in low or "gpt5.1" in low:
        return "GPT-5.1"
    if "gpt-4o-mini" in low or "gpt4o-mini" in low:
        return "GPT-4o-mini"
    if "gpt-4o" in low or "gpt4o" in low:
        return "GPT-4o"
    if "qwen" in low and "max" in low:
        return "Qwen-Max"
    if "qwen3" in low and "8b" in low:
        return "Qwen3-8B"
    if "llama" in low and "70b" in low:
        return "Llama 3.1 70B"
    if "llama" in low and "8b" in low:
        return "Llama 3.1 8B"
    return raw


def safe_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def prediction_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if re.fullmatch(r"pred_run_\d+", c)]
    return sorted(cols, key=lambda c: int(c.rsplit("_", 1)[1]))


def norm_ab(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"[^AB]", "", regex=True)
    )


def accuracy(df: pd.DataFrame, pred_col: str) -> float:
    if df.empty:
        return float("nan")
    return float((norm_ab(df[pred_col]) == norm_ab(df["gold_ans"])).mean())


def summarize_slice(df: pd.DataFrame, run_cols: list[str]) -> dict[str, float]:
    majority = accuracy(df, "prediction")
    per_run = [accuracy(df, c) for c in run_cols]

    if per_run:
        repeat_mean = float(statistics.mean(per_run))
        repeat_sd = float(statistics.stdev(per_run)) if len(per_run) >= 2 else 0.0
    else:
        repeat_mean = float("nan")
        repeat_sd = float("nan")

    return {
        "majority_accuracy": majority,
        "repeat_mean_accuracy": repeat_mean,
        "repeat_sd_sample": repeat_sd,
    }


def classify_condition(folder_name: str) -> str | None:
    if "__limitall__" not in folder_name:
        return None

    if folder_name.startswith("balanced__") and "__no_rag__" in folder_name:
        return "balanced_no_rag"

    if folder_name.startswith("balanced__") and "__rag__" in folder_name:
        return "balanced_rag"

    if (
        folder_name.startswith("pseudo_paper_final_model_ready__")
        and "__no_rag__" in folder_name
    ):
        return "pseudosentences_no_rag"

    return None


def model_from_folder_or_config(folder: Path) -> str:
    cfg = safe_json(folder / "config.json")
    raw = str(cfg.get("model", ""))

    if not raw:
        marker = "__model"
        if marker in folder.name:
            raw = folder.name.split(marker, 1)[1]

    return canonical_model(raw)


def add_slice(
    row: dict[str, Any],
    prefix: str,
    df: pd.DataFrame,
    run_cols: list[str],
) -> None:
    stats = summarize_slice(df, run_cols)
    for key, value in stats.items():
        row[f"{prefix}_{key}"] = value
    row[f"{prefix}_n"] = int(len(df))


def summarize_folder(folder: Path, condition: str) -> dict[str, Any] | None:
    pred_path = folder / "predictions.csv"
    if not pred_path.exists():
        return None

    df = pd.read_csv(pred_path)
    required = {"gold_ans", "prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{pred_path}: missing columns {sorted(missing)}")

    run_cols = prediction_columns(df)
    if not run_cols:
        raise ValueError(f"{pred_path}: no pred_run_N columns found")

    model = model_from_folder_or_config(folder)

    row: dict[str, Any] = {
        "condition": condition,
        "model": model,
        "folder": folder.name,
        "evaluation_rows": int(len(df)),
        "base_items": (
            int(df["source_idx"].nunique())
            if "source_idx" in df.columns
            else int(len(df))
        ),
        "repeats": int(len(run_cols)),
    }

    add_slice(row, "overall", df, run_cols)

    if "gold_scope_label" in df.columns:
        scope = df["gold_scope_label"].astype(str).str.strip().str.lower()
        add_slice(row, "surface", df.loc[scope == "surface"], run_cols)
        add_slice(row, "inverse", df.loc[scope == "inverse"], run_cols)

    if "comb" in df.columns:
        comb_num = pd.to_numeric(df["comb"], errors="coerce")
        for comb in [1, 2, 3, 4]:
            add_slice(row, f"comb_{comb}", df.loc[comb_num == comb], run_cols)

    return row


def print_compact_table(summary: pd.DataFrame, condition: str) -> None:
    sub = summary[summary["condition"] == condition].copy()
    if sub.empty:
        return

    order = {name: i for i, name in enumerate(FINAL_MODELS)}
    sub["_order"] = sub["model"].map(order).fillna(999)
    sub = sub.sort_values(["_order", "model"])

    print()
    print(condition)
    print("=" * len(condition))
    for _, r in sub.iterrows():
        overall = r.get("overall_majority_accuracy", float("nan"))
        sd = r.get("overall_repeat_sd_sample", float("nan"))
        surface = r.get("surface_majority_accuracy", float("nan"))
        inverse = r.get("inverse_majority_accuracy", float("nan"))
        print(
            f"{r['model']:<16} "
            f"overall={overall:.3f} ± {sd:.3f}  "
            f"surface={surface:.3f}  inverse={inverse:.3f}  "
            f"rows={int(r['evaluation_rows'])}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument(
        "--output",
        default="results/paper_metrics_summary.csv",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    if not results_root.exists():
        raise FileNotFoundError(f"Results directory not found: {results_root}")

    rows: list[dict[str, Any]] = []

    for folder in sorted(p for p in results_root.iterdir() if p.is_dir()):
        condition = classify_condition(folder.name)
        if condition is None:
            continue

        row = summarize_folder(folder, condition)
        if row is None:
            continue

        # Keep the final seven paper models in the paper-oriented summary.
        if row["model"] not in FINAL_MODELS:
            continue

        rows.append(row)

    if not rows:
        raise RuntimeError(
            "No final paper result folders found. Expected limitall balanced "
            "no-RAG/RAG and/or pseudo_paper_final_model_ready folders."
        )

    summary = pd.DataFrame(rows)

    condition_order = {
        "balanced_no_rag": 0,
        "pseudosentences_no_rag": 1,
        "balanced_rag": 2,
    }
    model_order = {name: i for i, name in enumerate(FINAL_MODELS)}

    summary["_condition_order"] = summary["condition"].map(condition_order)
    summary["_model_order"] = summary["model"].map(model_order)
    summary = summary.sort_values(
        ["_condition_order", "_model_order", "model"]
    ).drop(columns=["_condition_order", "_model_order"])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)

    for condition in condition_order:
        print_compact_table(summary, condition)

    print()
    print(f"[OK] Wrote: {out}")
    print(
        "[NOTE] Point estimates above are majority-vote accuracies; "
        "the ± value is sample SD across repeat-level accuracies."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
