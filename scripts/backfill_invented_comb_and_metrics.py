#!/usr/bin/env python3
"""
Backfill `comb` into invented artifacts and recompute per-combination metrics.

What this script does (one-off migration):
1) Ensures `data/private/invented_for_llms.csv` contains a `comb` column.
   - Prefer merging `comb` from `data/private/WB_survey_expB2_ABexpanded.csv` by `idx` if available.
   - Otherwise infer `comb` using a mapping learned from `data/private/dataset_for_llms.csv`
     (balanced dataset is treated as the canonical reference).

2) Iterates over `results/` subdirectories whose name starts with "invented":
   - updates `predictions.csv` by adding/filling `comb` (merge by `idx`),
   - rewrites `metrics.json` and `metrics.csv` so they include `accuracy_by_comb` (and `n_by_comb` in JSON),
     consistent with how `llm_zero_shot.py` computes metrics (aggregate by `idx` across repeats).

Safety:
- Makes timestamped .bak copies of files before overwriting them.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


# ----------------------------
# Helpers: IO + backups
# ----------------------------

def ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def backup_file(path: Path) -> Optional[Path]:
    """Create a timestamped backup next to the file. Return backup path or None if file does not exist."""
    if not path.exists():
        return None
    bak = path.with_suffix(path.suffix + f".bak-{ts()}")
    shutil.copy2(path, bak)
    return bak


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ----------------------------
# Comb mapping / inference
# ----------------------------

@dataclass(frozen=True)
class Paths:
    repo_root: Path
    results_dir: Path
    invented_for_llms: Path
    balanced_for_llms: Path
    wb_survey: Path


def _ensure_int_idx(df: pd.DataFrame, col: str = "idx") -> pd.DataFrame:
    if col not in df.columns:
        raise RuntimeError(f"Missing required column '{col}'. Columns: {list(df.columns)}")
    # Use pandas nullable integer, tolerate floats/strings in CSV
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def learn_comb_mapping_from_balanced(balanced_df: pd.DataFrame) -> Dict[Tuple[str, str, str], int]:
    """
    Learn a deterministic mapping:
      (OP1_type, OP2_type, gold_scope_label) -> comb
    from the balanced dataset (canonical).
    """
    required = ["OP1_type", "OP2_type", "gold_scope_label", "comb"]
    missing = [c for c in required if c not in balanced_df.columns]
    if missing:
        raise RuntimeError(f"Balanced dataset is missing columns: {missing}")

    # Normalize
    tmp = balanced_df[required].copy()
    tmp["OP1_type"] = tmp["OP1_type"].astype(str).str.strip()
    tmp["OP2_type"] = tmp["OP2_type"].astype(str).str.strip()
    tmp["gold_scope_label"] = tmp["gold_scope_label"].astype(str).str.strip()
    tmp["comb"] = pd.to_numeric(tmp["comb"], errors="coerce").astype("Int64")

    tmp = tmp.dropna(subset=["comb", "OP1_type", "OP2_type", "gold_scope_label"])
    if tmp.empty:
        raise RuntimeError("Balanced dataset has no usable rows to learn comb mapping.")

    # Detect conflicts: same key mapping to multiple comb values
    conflicts = (
        tmp.groupby(["OP1_type", "OP2_type", "gold_scope_label"])["comb"]
        .nunique()
        .reset_index()
        .query("comb > 1")
    )
    if not conflicts.empty:
        raise RuntimeError(
            "Conflicting comb mapping in balanced dataset. "
            "The same (OP1_type, OP2_type, gold_scope_label) maps to multiple comb values."
        )

    mapping: Dict[Tuple[str, str, str], int] = {}
    for (op1, op2, scope), grp in tmp.groupby(["OP1_type", "OP2_type", "gold_scope_label"]):
        comb_val = int(grp["comb"].iloc[0])
        mapping[(op1, op2, scope)] = comb_val

    # Optional sanity: expect combos 1..4 exist in mapping
    combos = sorted(set(mapping.values()))
    if combos and (min(combos) < 1 or max(combos) > 4):
        print(f"[WARN] Learned comb values outside 1..4: {combos}")

    return mapping


def add_or_fill_comb_in_invented(
    invented_df: pd.DataFrame,
    *,
    wb_df: Optional[pd.DataFrame],
    balanced_mapping: Dict[Tuple[str, str, str], int],
) -> pd.DataFrame:
    """
    Ensure invented_df has comb:
    - Prefer merge from WB by idx if WB has comb.
    - Otherwise infer from balanced_mapping based on (OP1_type, OP2_type, gold_scope_label).
    """
    invented_df = invented_df.copy()
    invented_df = _ensure_int_idx(invented_df, "idx")

    # Merge from WB if available and has comb
    if wb_df is not None and "comb" in wb_df.columns and "idx" in wb_df.columns:
        wb = wb_df.copy()
        wb = _ensure_int_idx(wb, "idx")
        wb["comb"] = pd.to_numeric(wb["comb"], errors="coerce").astype("Int64")
        wb = wb.dropna(subset=["idx", "comb"])[["idx", "comb"]].drop_duplicates("idx")

        if not wb["idx"].is_unique:
            raise RuntimeError("WB survey idx is not unique after deduplication; cannot safely merge comb.")

        before_missing = invented_df["comb"].isna().sum() if "comb" in invented_df.columns else len(invented_df)

        if "comb" not in invented_df.columns:
            invented_df = invented_df.merge(wb, on="idx", how="left")
        else:
            invented_df["comb"] = pd.to_numeric(invented_df["comb"], errors="coerce").astype("Int64")
            invented_df = invented_df.merge(wb, on="idx", how="left", suffixes=("", "_wb"))
            if "comb_wb" in invented_df.columns:
                invented_df["comb"] = invented_df["comb"].combine_first(invented_df["comb_wb"])
                invented_df = invented_df.drop(columns=["comb_wb"])

        after_missing = invented_df["comb"].isna().sum()
        print(f"[invented_for_llms] comb filled from WB by idx. Missing: {before_missing} -> {after_missing}")

    # If still missing, infer via mapping from balanced
    if "comb" not in invented_df.columns:
        invented_df["comb"] = pd.NA

    invented_df["comb"] = pd.to_numeric(invented_df["comb"], errors="coerce").astype("Int64")

    missing_mask = invented_df["comb"].isna()
    if missing_mask.any():
        required = ["OP1_type", "OP2_type", "gold_scope_label"]
        missing_cols = [c for c in required if c not in invented_df.columns]
        if missing_cols:
            raise RuntimeError(
                f"Cannot infer comb (missing columns {missing_cols}) and WB merge did not fill it."
            )

        def infer_row(row: pd.Series) -> Optional[int]:
            key = (
                str(row["OP1_type"]).strip(),
                str(row["OP2_type"]).strip(),
                str(row["gold_scope_label"]).strip(),
            )
            return balanced_mapping.get(key)

        inferred = invented_df.loc[missing_mask, :].apply(infer_row, axis=1)
        invented_df.loc[missing_mask, "comb"] = inferred.values

    # Final validation
    still_missing = invented_df["comb"].isna().sum()
    if still_missing:
        sample = invented_df[invented_df["comb"].isna()][["idx", "OP1_type", "OP2_type", "gold_scope_label"]].head(10)
        raise RuntimeError(
            f"Still missing comb for {still_missing} rows in invented_for_llms.\n"
            f"Sample rows:\n{sample.to_string(index=False)}\n"
            "Fix mapping or ensure WB_survey_expB2_ABexpanded.csv contains comb for those idx."
        )

    # Normalize to plain int in CSV (still written as numbers)
    invented_df["comb"] = invented_df["comb"].astype("Int64")
    return invented_df


# ----------------------------
# Metrics recomputation (match llm_zero_shot logic)
# ----------------------------

def compute_metrics_like_llm_zero_shot(pred: pd.DataFrame) -> dict:
    """
    Recompute metrics in the same spirit as src/llm_zero_shot.py:
    - accuracy_overall
    - accuracy_by_gold_scope_label (if available)
    - accuracy_by_comb (if available)
    NOTE: This function is robust to different correctness column names:
          'correct' (current), 'is_correct' (legacy), or no correctness column at all
          (then it tries to compute correctness from prediction vs gold).
    """

    pred = pred.copy()

    # 1) Ensure we have a boolean 'correct' column.
    if "correct" not in pred.columns:
        if "is_correct" in pred.columns:
            pred = pred.rename(columns={"is_correct": "correct"})
        else:
            # Try to compute 'correct' from prediction vs gold.
            pred_col_candidates = ["prediction", "pred", "answer", "choice"]
            gold_col_candidates = ["gold_ans", "gold", "label", "gold_label", "gold_answer"]

            pred_col = next((c for c in pred_col_candidates if c in pred.columns), None)
            gold_col = next((c for c in gold_col_candidates if c in pred.columns), None)

            if pred_col is None or gold_col is None:
                raise RuntimeError(
                    "predictions.csv missing correctness info. "
                    "Expected 'correct' or 'is_correct' or at least a pair of columns like "
                    "'prediction' + 'gold_ans'. "
                    f"Found columns: {list(pred.columns)}"
                )

            pred_vals = pred[pred_col].astype(str).str.strip().str.upper()
            gold_vals = pred[gold_col].astype(str).str.strip().str.upper()

            # Treat invalid / empty predictions as incorrect (False), consistent with how
            # comparisons behave in llm_zero_shot when prediction != gold.
            valid_pred = pred_vals.isin(["A", "B"])
            valid_gold = gold_vals.isin(["A", "B"])

            pred["correct"] = (valid_pred & valid_gold & (pred_vals == gold_vals))

    # Normalize dtype
    pred["correct"] = pred["correct"].astype(bool)

    # Helper identical in spirit to llm_zero_shot.compute_group_accuracy
    def group_acc(df_group: pd.DataFrame) -> dict:
        return {"accuracy": float(df_group["correct"].mean()), "n": int(len(df_group))}

    metrics: dict = {}
    metrics["n"] = int(len(pred))
    metrics["accuracy_overall"] = float(pred["correct"].mean())

    if "gold_scope_label" in pred.columns:
        metrics["accuracy_by_gold_scope_label"] = {
            str(k): group_acc(g) for k, g in pred.groupby("gold_scope_label", dropna=False)
        }

    if "comb" in pred.columns:
        acc_by_comb = {}
        for k, g in pred.groupby("comb", dropna=False):
            if pd.isna(k):
                continue
            # keep keys exactly as strings: "1".."4"
            try:
                key = str(int(k))
            except Exception:
                key = str(k)
            acc_by_comb[key] = group_acc(g)
        metrics["accuracy_by_comb"] = acc_by_comb

    return metrics


def metrics_to_long_csv(metrics: Dict) -> pd.DataFrame:
    """
    Create metrics.csv in the same "long" format as the project uses:
      metric,value,n

    Order:
    - accuracy
    - accuracy_by_scope_label::inverse
    - accuracy_by_scope_label::surface
    - remaining scope labels (stable order)
    - accuracy_by_comb::1..4
    """
    rows: List[Dict] = []
    rows.append({"metric": "accuracy", "value": metrics.get("accuracy", 0.0), "n": metrics.get("n", 0)})

    # Scope label rows
    acc_scope = metrics.get("accuracy_by_scope_label", {}) or {}
    n_scope = metrics.get("n_by_scope_label", {}) or {}

    preferred_scope_order = ["inverse", "surface"]
    seen = set()

    for lab in preferred_scope_order:
        if lab in acc_scope:
            rows.append(
                {"metric": f"accuracy_by_scope_label::{lab}", "value": acc_scope[lab], "n": n_scope.get(lab, 0)}
            )
            seen.add(lab)

    for lab in sorted(k for k in acc_scope.keys() if k not in seen):
        rows.append({"metric": f"accuracy_by_scope_label::{lab}", "value": acc_scope[lab], "n": n_scope.get(lab, 0)})

    # Comb rows (force 1..4 for downstream summaries)
    acc_comb = metrics.get("accuracy_by_comb", {}) or {}
    n_comb = metrics.get("n_by_comb", {}) or {}

    for c in ["1", "2", "3", "4"]:
        rows.append(
            {
                "metric": f"accuracy_by_comb::{c}",
                "value": acc_comb.get(c, float("nan")),
                "n": n_comb.get(c, 0),
            }
        )

    return pd.DataFrame(rows, columns=["metric", "value", "n"])


# ----------------------------
# Main migration loop
# ----------------------------

def update_predictions_with_comb(pred_path: Path, invented_df: pd.DataFrame) -> pd.DataFrame:
    pred = read_csv(pred_path)
    pred = _ensure_int_idx(pred, "idx")

    if not invented_df["idx"].is_unique:
        raise RuntimeError("invented_for_llms idx is not unique; cannot merge comb reliably.")

    lookup = invented_df[["idx", "comb"]].copy()
    lookup["comb"] = pd.to_numeric(lookup["comb"], errors="coerce").astype("Int64")

    # Merge + fill
    if "comb" not in pred.columns:
        pred = pred.merge(lookup, on="idx", how="left")
    else:
        pred["comb"] = pd.to_numeric(pred["comb"], errors="coerce").astype("Int64")
        pred = pred.merge(lookup, on="idx", how="left", suffixes=("", "_ds"))
        if "comb_ds" in pred.columns:
            pred["comb"] = pred["comb"].combine_first(pred["comb_ds"])
            pred = pred.drop(columns=["comb_ds"])

    missing = pred["comb"].isna().sum()
    if missing:
        sample = pred[pred["comb"].isna()][["idx"]].head(10)
        raise RuntimeError(
            f"After merge, predictions still miss comb for {missing} rows in {pred_path}.\n"
            f"Sample idx:\n{sample.to_string(index=False)}"
        )

    pred["comb"] = pred["comb"].astype("Int64")
    return pred


def process_invented_result_dir(result_dir: Path, invented_df: pd.DataFrame) -> None:
    pred_path = result_dir / "predictions.csv"
    metrics_json_path = result_dir / "metrics.json"
    metrics_csv_path = result_dir / "metrics.csv"

    if not pred_path.exists():
        print(f"[SKIP] {result_dir} (no predictions.csv)")
        return

    # Update predictions.csv
    backup_file(pred_path)
    pred = update_predictions_with_comb(pred_path, invented_df)
    write_csv(pred, pred_path)

    # Recompute metrics (and merge into existing metrics.json if present)
    new_metrics = compute_metrics_like_llm_zero_shot(pred)

    existing_metrics: Dict = {}
    if metrics_json_path.exists():
        backup_file(metrics_json_path)
        try:
            existing_metrics = read_json(metrics_json_path)
        except Exception:
            existing_metrics = {}

    merged_metrics = dict(existing_metrics)
    merged_metrics.update(new_metrics)
    write_json(merged_metrics, metrics_json_path)

    # Rewrite metrics.csv in the "long" format
    backup_file(metrics_csv_path)
    long_df = metrics_to_long_csv(merged_metrics)
    write_csv(long_df, metrics_csv_path)

    print(f"[OK] Updated: {result_dir.name} (predictions.csv + metrics.json + metrics.csv)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default=".", help="Path to the repository root (default: .)")
    ap.add_argument("--results-dir", type=str, default="results", help="Results directory (default: results)")
    ap.add_argument(
        "--invented-for-llms",
        type=str,
        default="data/private/invented_for_llms.csv",
        help="Invented dataset used by LLMs (default: data/private/invented_for_llms.csv)",
    )
    ap.add_argument(
        "--balanced-for-llms",
        type=str,
        default="data/private/dataset_for_llms.csv",
        help="Balanced dataset used by LLMs (canonical mapping source; default: data/private/dataset_for_llms.csv)",
    )
    ap.add_argument(
        "--wb-survey",
        type=str,
        default="data/private/WB_survey_expB2_ABexpanded.csv",
        help="WB survey CSV (if it has comb, we can merge by idx; default: data/private/WB_survey_expB2_ABexpanded.csv)",
    )
    ap.add_argument(
        "--only",
        type=str,
        choices=["all", "dataset", "results"],
        default="all",
        help="What to update: dataset=only invented_for_llms, results=only results/*, all=both",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    paths = Paths(
        repo_root=repo_root,
        results_dir=(repo_root / args.results_dir),
        invented_for_llms=(repo_root / args.invented_for_llms),
        balanced_for_llms=(repo_root / args.balanced_for_llms),
        wb_survey=(repo_root / args.wb_survey),
    )

    # Load balanced mapping
    if not paths.balanced_for_llms.exists():
        raise RuntimeError(f"Missing balanced_for_llms: {paths.balanced_for_llms}")
    balanced_df = read_csv(paths.balanced_for_llms)
    comb_mapping = learn_comb_mapping_from_balanced(balanced_df)

    # Load WB (optional)
    wb_df: Optional[pd.DataFrame] = None
    if paths.wb_survey.exists():
        try:
            wb_df = read_csv(paths.wb_survey)
        except Exception:
            wb_df = None

    # Update invented_for_llms.csv
    if args.only in ("all", "dataset"):
        if not paths.invented_for_llms.exists():
            raise RuntimeError(f"Missing invented_for_llms: {paths.invented_for_llms}")

        inv_df = read_csv(paths.invented_for_llms)
        inv_df = add_or_fill_comb_in_invented(inv_df, wb_df=wb_df, balanced_mapping=comb_mapping)

        backup_file(paths.invented_for_llms)
        write_csv(inv_df, paths.invented_for_llms)
        print(f"[OK] Updated dataset: {paths.invented_for_llms}")

    # Reload invented df (with comb) for results update
    inv_df = read_csv(paths.invented_for_llms)
    inv_df = _ensure_int_idx(inv_df, "idx")
    if "comb" not in inv_df.columns:
        raise RuntimeError("invented_for_llms still missing comb after update.")
    inv_df["comb"] = pd.to_numeric(inv_df["comb"], errors="coerce").astype("Int64")
    if inv_df["comb"].isna().any():
        raise RuntimeError("invented_for_llms has NaN comb values; cannot proceed.")

    # Update results directories
    if args.only in ("all", "results"):
        if not paths.results_dir.exists():
            raise RuntimeError(f"Missing results dir: {paths.results_dir}")

        invented_dirs = sorted([p for p in paths.results_dir.iterdir() if p.is_dir() and p.name.startswith("invented")])
        if not invented_dirs:
            print("[WARN] No results subdirectories starting with 'invented' were found. Nothing to do.")
            return

        print(f"[INFO] Found {len(invented_dirs)} invented result dirs.")
        for d in invented_dirs:
            process_invented_result_dir(d, inv_df)

    print("[DONE] Backfill completed successfully.")


if __name__ == "__main__":
    main()
