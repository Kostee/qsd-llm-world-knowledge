#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REQUIRED_COLUMNS = [
    "comb",
    "sentence",
    "pseudo_sentence",
    "Option A",
    "pseudo_Option_A",
    "Option B",
    "pseudo_Option_B",
    "gold_ans",
    "gold_scope_label",
]


def validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    null_counts = df[REQUIRED_COLUMNS].isna().sum()
    bad_nulls = {k: int(v) for k, v in null_counts.items() if int(v) > 0}
    if bad_nulls:
        raise ValueError(f"Found missing values in required columns: {bad_nulls}")

    gold_values = set(df["gold_ans"].astype(str).str.strip().unique())
    if not gold_values.issubset({"A", "B"}):
        raise ValueError(f"Unexpected gold_ans values: {sorted(gold_values)}")


def build_output(df: pd.DataFrame) -> pd.DataFrame:
    src = df.copy().reset_index(drop=True)
    src.insert(0, "source_idx", range(1, len(src) + 1))

    base = pd.DataFrame(
        {
            "source_idx": src["source_idx"],
            "sentence": src["pseudo_sentence"],
            "Option A": src["pseudo_Option_A"],
            "Option B": src["pseudo_Option_B"],
            "gold_ans": src["gold_ans"].astype(str).str.strip(),
            "gold_scope_label": src["gold_scope_label"],
            "comb": src["comb"],
            "english_sentence": src["sentence"],
            "english_Option_A": src["Option A"],
            "english_Option_B": src["Option B"],
            "order_variant": "original",
        }
    )

    flipped = base.copy()
    flipped[["Option A", "Option B"]] = flipped[["Option B", "Option A"]]
    flipped["gold_ans"] = flipped["gold_ans"].map({"A": "B", "B": "A"})
    flipped["order_variant"] = "flipped"

    final = pd.concat([base, flipped], ignore_index=True)
    final.insert(0, "idx", range(1, len(final) + 1))

    return final


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the final pseudo dataset for EMNLP reruns by taking pseudo columns "
            "from balanced_dataset_only_A_v2.csv and appending A/B-swapped duplicates."
        )
    )
    parser.add_argument(
        "--input",
        default="data/private/balanced_dataset_only_A_v2.csv",
        help="Path to the converted CSV from Justyna (default: data/private/balanced_dataset_only_A_v2.csv)",
    )
    parser.add_argument(
        "--output",
        default="data/private/pseudo_for_llms_emnlp26.csv",
        help="Output CSV path (default: data/private/pseudo_for_llms_emnlp26.csv)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    validate_input(df)
    final = build_output(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path, index=False)

    print(f"Wrote: {output_path}")
    print(f"Input rows:  {len(df)}")
    print(f"Output rows: {len(final)}")
    print("gold_ans counts:")
    print(final["gold_ans"].value_counts().sort_index().to_string())
    print("comb counts:")
    print(final["comb"].value_counts().sort_index().to_string())
    print("order_variant counts:")
    print(final["order_variant"].value_counts().sort_index().to_string())

    expected_rows = len(df) * 2
    if len(final) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} output rows, got {len(final)}")

    if set(final["gold_ans"].unique()) != {"A", "B"}:
        raise AssertionError("Expected both A and B in gold_ans after flipping")

    if any(final[["sentence", "Option A", "Option B", "gold_ans", "gold_scope_label", "comb"]].isna().sum()):
        raise AssertionError("Found missing values in final required fields")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
