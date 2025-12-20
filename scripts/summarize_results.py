#!/usr/bin/env python3
"""
Create a flat metrics summary CSV from results/* runs (only limit=all).

Input per run directory:
  results/<run_dir>/
    - config.json (optional)
    - metrics.json (optional but expected)
    - predictions.csv (optional)

Output:
  results/metrics_summary.csv
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class RunSummary:
    model: Optional[str]
    provider: Optional[str]
    dataset: Optional[str]
    rag: Optional[bool]
    repeats: Optional[int]
    n_items: Optional[int]

    accuracy_overall: Optional[float]
    accuracy_surface: Optional[float]
    accuracy_inverse: Optional[float]

    accuracy_comb_1: Optional[float]
    accuracy_comb_2: Optional[float]
    accuracy_comb_3: Optional[float]
    accuracy_comb_4: Optional[float]

    runtime_seconds_total: Optional[float]


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[WARN] Failed to read JSON {path}: {e}", file=sys.stderr)
        return None


def _count_items_from_predictions(pred_path: Path) -> Optional[int]:
    if not pred_path.exists():
        return None
    try:
        with pred_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return 0
            return sum(1 for _ in reader)
    except Exception:
        try:
            df = pd.read_csv(pred_path)
            return int(len(df))
        except Exception:
            return None


def _infer_provider(model_name: Optional[str]) -> Optional[str]:
    if not model_name:
        return None
    m = model_name.lower()
    if m.startswith("qwen:"):
        return "qwen"
    if m.startswith("gemini:"):
        return "gemini"
    if m.startswith("openrouter:"):
        return "openrouter"
    return "openai"


def _round(v: Optional[float], ndigits: int = 4) -> Optional[float]:
    if v is None:
        return None
    try:
        return round(float(v), ndigits)
    except Exception:
        return None


def _reconstruct_model_from_safe(safe: str) -> str:
    """
    Folder-safe -> actual model heuristic:
      - qwen_qwen3-8b -> qwen:qwen3-8b
      - gemini_gemini-2.0-flash-exp -> gemini:gemini-2.0-flash-exp
      - openrouter_meta-llama_llama-3.1-8b-instruct -> openrouter:meta-llama/llama-3.1-8b-instruct
      - gpt-4o -> gpt-4o
    """
    if safe.startswith("qwen_"):
        return "qwen:" + safe[len("qwen_") :]
    if safe.startswith("gemini_"):
        return "gemini:" + safe[len("gemini_") :]
    if safe.startswith("openrouter_"):
        rest = safe[len("openrouter_") :]
        # restore first "/" only (good for meta-llama/<model> pattern)
        if "_" in rest:
            rest = rest.replace("_", "/", 1)
        return "openrouter:" + rest
    return safe


def _parse_run_dir_name(run_dir_name: str) -> Dict[str, Any]:
    """
    Example:
    balanced__limitall__no_rag__repeats5__seed42__modelgpt-4o
    """
    parts = run_dir_name.split("__")
    out: Dict[str, Any] = {
        "dataset": None,
        "limit": None,
        "rag": None,
        "repeats": None,
        "model": None,
    }

    if parts:
        out["dataset"] = parts[0]  # balanced / invented (usually)

    for p in parts[1:]:
        if p.startswith("limit"):
            out["limit"] = p.replace("limit", "", 1) or None
        elif p == "rag":
            out["rag"] = True
        elif p == "no_rag":
            out["rag"] = False
        elif p.startswith("repeats"):
            try:
                out["repeats"] = int(p.replace("repeats", "", 1))
            except Exception:
                pass
        elif p.startswith("model"):
            safe = p.replace("model", "", 1)
            out["model"] = _reconstruct_model_from_safe(safe)

    return out


def _is_limit_all(run_dir_name: str, config: Optional[Dict[str, Any]]) -> bool:
    if "__limitall__" in run_dir_name:
        return True
    if config:
        lim = config.get("limit")
        if isinstance(lim, str) and lim.lower() == "all":
            return True
    return False


def _get_accuracy_surface_inverse(metrics: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    by_scope = metrics.get("accuracy_by_gold_scope_label") or {}
    surface = None
    inverse = None
    try:
        surface = by_scope.get("surface", {}).get("accuracy")
    except Exception:
        surface = None
    try:
        inverse = by_scope.get("inverse", {}).get("accuracy")
    except Exception:
        inverse = None
    return surface, inverse


def _get_accuracy_by_comb(metrics: Dict[str, Any]) -> Dict[str, Optional[float]]:
    by_comb = metrics.get("accuracy_by_comb") or {}
    out = {}
    for k in ["1", "2", "3", "4"]:
        try:
            out[k] = by_comb.get(k, {}).get("accuracy")
        except Exception:
            out[k] = None
    return out


def summarize_results(results_root: Path) -> List[RunSummary]:
    rows: List[RunSummary] = []

    for run_dir in sorted([p for p in results_root.iterdir() if p.is_dir()]):
        config = _safe_read_json(run_dir / "config.json")
        metrics = _safe_read_json(run_dir / "metrics.json") or {}
        pred_path = run_dir / "predictions.csv"

        if not _is_limit_all(run_dir.name, config):
            continue

        parsed = _parse_run_dir_name(run_dir.name)

        # Prefer config; fallback to parsed-from-folder
        dataset = (config or {}).get("dataset") or parsed.get("dataset")
        rag = (config or {}).get("rag")
        if rag is None:
            rag = parsed.get("rag")
        model = (config or {}).get("model") or parsed.get("model")
        repeats = (config or {}).get("repeats") or parsed.get("repeats")

        provider = _infer_provider(model)

        # n_items
        n_items = None
        if "n" in metrics:
            try:
                n_items = int(metrics["n"])
            except Exception:
                n_items = None
        if n_items is None:
            n_items = _count_items_from_predictions(pred_path)

        # accuracies
        acc_overall = metrics.get("accuracy_overall")
        surface, inverse = _get_accuracy_surface_inverse(metrics)
        by_comb = _get_accuracy_by_comb(metrics)

        # runtime
        runtime_seconds = metrics.get("runtime_seconds_total") or metrics.get("runtime_seconds")

        rows.append(
            RunSummary(
                model=str(model) if model is not None else None,
                provider=provider,
                dataset=str(dataset) if dataset is not None else None,
                rag=bool(rag) if rag is not None else None,
                repeats=int(repeats) if repeats is not None else None,
                n_items=int(n_items) if n_items is not None else None,
                accuracy_overall=_round(acc_overall),
                accuracy_surface=_round(surface),
                accuracy_inverse=_round(inverse),
                accuracy_comb_1=_round(by_comb.get("1")),
                accuracy_comb_2=_round(by_comb.get("2")),
                accuracy_comb_3=_round(by_comb.get("3")),
                accuracy_comb_4=_round(by_comb.get("4")),
                runtime_seconds_total=_round(runtime_seconds, ndigits=2) if runtime_seconds is not None else None,
            )
        )

    return rows


def main() -> None:
    repo_root = Path.cwd()
    results_root = repo_root / "results"
    if not results_root.exists():
        print(f"[ERROR] results/ not found at: {results_root}", file=sys.stderr)
        sys.exit(1)

    rows = summarize_results(results_root)
    if not rows:
        print("[INFO] No limit=all runs found.")
        return

    df = pd.DataFrame([asdict(r) for r in rows])

    # Enforce column order (as requested)
    col_order = [
        "model",
        "provider",
        "dataset",
        "rag",
        "repeats",
        "n_items",
        "accuracy_overall",
        "accuracy_surface",
        "accuracy_inverse",
        "accuracy_comb_1",
        "accuracy_comb_2",
        "accuracy_comb_3",
        "accuracy_comb_4",
        "runtime_seconds_total",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    # Make n_items a proper integer column (nullable)
    if "n_items" in df.columns:
        df["n_items"] = pd.to_numeric(df["n_items"], errors="coerce").astype("Int64")

    out_csv = results_root / "metrics_summary.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"[DONE] Wrote: {out_csv}")
    print(f"[DONE] Rows: {len(df)}")


if __name__ == "__main__":
    main()
