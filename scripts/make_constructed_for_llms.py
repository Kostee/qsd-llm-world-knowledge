#!/usr/bin/env python3
"""
Create a deterministic single-choice dataset for LLM evaluation by selecting
exactly one row per pair (grouped by `idx`) from the invented dataset.

Expected input format (as in WB_survey_expB2_ABexpanded.csv):
- `idx` identifies the pair id
- each idx occurs exactly twice (A/B swapped)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input CSV (e.g., data/private/WB_survey_expB2_ABexpanded.csv)",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output CSV (e.g., data/private/invented_for_llms.csv)",
    )
    p.add_argument(
        "--group-col",
        type=str,
        default="idx",
        help="Column that identifies the pair/group (default: idx)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic selection (default: 42)",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any group does not have exactly 2 rows (recommended).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    if args.group_col not in df.columns:
        raise ValueError(
            f"Group column '{args.group_col}' not found. Columns: {list(df.columns)}"
        )

    # Check group sizes
    group_sizes = df.groupby(args.group_col).size()
    bad = group_sizes[group_sizes != 2]
    if len(bad) > 0:
        msg = (
            f"Found {len(bad)} groups with size != 2. "
            f"Examples: {bad.head(10).to_dict()}"
        )
        if args.strict:
            raise ValueError(msg)
        else:
            print(f"[WARN] {msg}")

    # Deterministic sample: pick 1 row per group
    sampled = (
        df.groupby(args.group_col, group_keys=False)
          .apply(lambda g: g.sample(n=1, random_state=args.seed))
          .reset_index(drop=True)
    )

    # Sort by idx for readability (optional)
    sampled = sampled.sort_values(by=args.group_col).reset_index(drop=True)

    sampled.to_csv(out_path, index=False)

    print(f"[OK] Input rows:  {len(df)}")
    print(f"[OK] Output rows: {len(sampled)} (one per {args.group_col})")
    print(f"[OK] Saved to:    {out_path}")


if __name__ == "__main__":
    main()