#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare the final EMNLP 2026 pseudosentence evaluation files.

Starting from the canonical 160-item pseudosentence dataset, this script:
1. validates the source benchmark,
2. creates an original and an A/B-flipped version of every item,
3. writes the 320-row model-ready evaluation dataset,
4. writes a prompt-only version without gold labels or metadata,
5. optionally verifies the generated files against historical files used
   in the reported experiments.

The English source sentences are intentionally not required by this script
and are not included in the public benchmark.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REQUIRED_SOURCE_COLUMNS = [
    "source_idx",
    "sentence",
    "Option A",
    "Option B",
    "gold_ans",
    "gold_scope_label",
    "comb",
]

MODEL_READY_COLUMNS = [
    "idx",
    "source_idx",
    "sentence",
    "Option A",
    "Option B",
    "gold_ans",
    "gold_scope_label",
    "comb",
    "order_variant",
]

PROMPT_ONLY_COLUMNS = [
    "sentence",
    "Option A",
    "Option B",
]

EXPECTED_ITEMS = 160
EXPECTED_ITEMS_PER_COMBINATION = 40


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def validate_source(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required source columns: {missing}")

    out = df[REQUIRED_SOURCE_COLUMNS].copy()

    # Check missing values before converting textual columns to strings.
    if out.isna().any().any():
        bad = out.isna().sum()
        bad = bad[bad > 0].to_dict()
        raise ValueError(f"Missing values in source dataset: {bad}")

    # Preserve stimulus and option text exactly as stored in the canonical
    # source dataset. This is required for exact reproduction of the
    # historical model input.
    for col in ["sentence", "Option A", "Option B"]:
        out[col] = out[col].astype(str)

    # Metadata labels may be safely normalized.
    out["gold_ans"] = out["gold_ans"].astype(str).str.strip().str.upper()
    out["gold_scope_label"] = (
        out["gold_scope_label"].astype(str).str.strip().str.lower()
    )

    out["comb"] = pd.to_numeric(out["comb"], errors="raise").astype(int)
    out["source_idx"] = pd.to_numeric(out["source_idx"], errors="raise").astype(int)

    if len(out) != EXPECTED_ITEMS:
        raise ValueError(
            f"Expected {EXPECTED_ITEMS} source items, found {len(out)}"
        )

    if not out["source_idx"].is_unique:
        duplicates = out.loc[
            out["source_idx"].duplicated(keep=False), "source_idx"
        ].tolist()
        raise ValueError(f"Duplicate source_idx values: {duplicates}")

    expected_ids = list(range(1, EXPECTED_ITEMS + 1))
    actual_ids = sorted(out["source_idx"].tolist())
    if actual_ids != expected_ids:
        raise ValueError(
            "source_idx must contain exactly 1..160 for the final benchmark"
        )

    # This canonical source intentionally places the gold interpretation in A.
    gold_values = set(out["gold_ans"].unique())
    if gold_values != {"A"}:
        raise ValueError(
            "The canonical 160-item source is expected to have gold_ans=A "
            f"for every item; found {sorted(gold_values)}"
        )

    scope_values = set(out["gold_scope_label"].unique())
    if scope_values != {"surface", "inverse"}:
        raise ValueError(
            f"Unexpected scope labels: {sorted(scope_values)}"
        )

    comb_values = set(out["comb"].unique())
    if comb_values != {1, 2, 3, 4}:
        raise ValueError(
            f"Expected combinations 1,2,3,4; found {sorted(comb_values)}"
        )

    comb_counts = out["comb"].value_counts().sort_index()
    expected_comb_counts = {
        1: EXPECTED_ITEMS_PER_COMBINATION,
        2: EXPECTED_ITEMS_PER_COMBINATION,
        3: EXPECTED_ITEMS_PER_COMBINATION,
        4: EXPECTED_ITEMS_PER_COMBINATION,
    }
    if comb_counts.to_dict() != expected_comb_counts:
        raise ValueError(
            "Unexpected combination counts: "
            f"{comb_counts.to_dict()} "
            f"(expected {expected_comb_counts})"
        )

    scope_counts = out["gold_scope_label"].value_counts().to_dict()
    if scope_counts != {"surface": 80, "inverse": 80}:
        raise ValueError(
            f"Unexpected surface/inverse counts: {scope_counts}"
        )

    # Combination definitions used in the paper.
    expected_scope_by_comb = {
        1: "surface",
        2: "surface",
        3: "inverse",
        4: "inverse",
    }

    for comb, expected_scope in expected_scope_by_comb.items():
        observed = set(
            out.loc[out["comb"] == comb, "gold_scope_label"].unique()
        )
        if observed != {expected_scope}:
            raise ValueError(
                f"Combination {comb} should map to {expected_scope}, "
                f"found {sorted(observed)}"
            )

    return out.sort_values("source_idx").reset_index(drop=True)


def build_model_ready(source: pd.DataFrame) -> pd.DataFrame:
    base = source.copy()
    base["order_variant"] = "original"

    flipped = source.copy()

    # Swap options positionally, avoiding pandas label-alignment surprises.
    flipped[["Option A", "Option B"]] = (
        source[["Option B", "Option A"]].to_numpy()
    )

    flipped["gold_ans"] = source["gold_ans"].map(
        {"A": "B", "B": "A"}
    )
    flipped["order_variant"] = "flipped"

    final = pd.concat([base, flipped], ignore_index=True)
    final.insert(0, "idx", range(1, len(final) + 1))

    return final[MODEL_READY_COLUMNS]


def build_prompt_only(model_ready: pd.DataFrame) -> pd.DataFrame:
    return model_ready[PROMPT_ONLY_COLUMNS].copy()


def validate_model_ready(df: pd.DataFrame) -> None:
    expected_rows = EXPECTED_ITEMS * 2

    if len(df) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} model-ready rows, found {len(df)}"
        )

    if list(df.columns) != MODEL_READY_COLUMNS:
        raise AssertionError(
            f"Unexpected model-ready columns: {list(df.columns)}"
        )

    variant_counts = df["order_variant"].value_counts().to_dict()
    if variant_counts != {"original": 160, "flipped": 160}:
        raise AssertionError(
            f"Unexpected order_variant counts: {variant_counts}"
        )

    gold_counts = df["gold_ans"].value_counts().to_dict()
    if gold_counts != {"A": 160, "B": 160}:
        raise AssertionError(
            f"Unexpected gold A/B counts: {gold_counts}"
        )

    comb_counts = df["comb"].value_counts().sort_index().to_dict()
    if comb_counts != {1: 80, 2: 80, 3: 80, 4: 80}:
        raise AssertionError(
            f"Unexpected combination counts: {comb_counts}"
        )

    scope_counts = df["gold_scope_label"].value_counts().to_dict()
    if scope_counts != {"surface": 160, "inverse": 160}:
        raise AssertionError(
            f"Unexpected scope counts: {scope_counts}"
        )

    # Every source item must occur exactly twice:
    # one original and one flipped.
    per_source = df.groupby("source_idx")

    if not (per_source.size() == 2).all():
        raise AssertionError(
            "Every source_idx must occur exactly twice"
        )

    for source_idx, pair in per_source:
        if set(pair["order_variant"]) != {"original", "flipped"}:
            raise AssertionError(
                f"source_idx={source_idx}: missing original/flipped pair"
            )

        original = pair[pair["order_variant"] == "original"].iloc[0]
        flipped = pair[pair["order_variant"] == "flipped"].iloc[0]

        if original["sentence"] != flipped["sentence"]:
            raise AssertionError(
                f"source_idx={source_idx}: sentence changed after flipping"
            )

        if original["Option A"] != flipped["Option B"]:
            raise AssertionError(
                f"source_idx={source_idx}: flipped Option B mismatch"
            )

        if original["Option B"] != flipped["Option A"]:
            raise AssertionError(
                f"source_idx={source_idx}: flipped Option A mismatch"
            )

        expected_flipped_gold = {
            "A": "B",
            "B": "A",
        }[original["gold_ans"]]

        if flipped["gold_ans"] != expected_flipped_gold:
            raise AssertionError(
                f"source_idx={source_idx}: gold answer was not flipped"
            )

        if original["gold_scope_label"] != flipped["gold_scope_label"]:
            raise AssertionError(
                f"source_idx={source_idx}: scope label changed after flipping"
            )

        if original["comb"] != flipped["comb"]:
            raise AssertionError(
                f"source_idx={source_idx}: combination changed after flipping"
            )


def verify_against(
    generated: pd.DataFrame,
    historical_path: Path | None,
    label: str,
) -> None:
    if historical_path is None:
        return

    historical = load_csv(historical_path)

    try:
        pd.testing.assert_frame_equal(
            generated.reset_index(drop=True),
            historical.reset_index(drop=True),
            check_dtype=False,
            check_like=False,
        )
    except AssertionError as exc:
        raise AssertionError(
            f"{label} does NOT match historical file "
            f"{historical_path}:\n{exc}"
        ) from exc

    print(f"[OK] {label} exactly matches historical file: {historical_path}")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[OK] Wrote: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the final 320-row EMNLP 2026 pseudosentence "
            "evaluation dataset from the canonical 160-item source."
        )
    )

    parser.add_argument(
        "--input",
        default="data/public/pseudosentences_emnlp2026.csv",
        help=(
            "Canonical 160-item source dataset "
            "(default: data/public/pseudosentences_emnlp2026.csv)"
        ),
    )

    parser.add_argument(
        "--model-ready-output",
        default="data/generated/pseudo_paper_final_model_ready.csv",
        help=(
            "Output path for the 320-row model-ready dataset "
            "(default: data/generated/pseudo_paper_final_model_ready.csv)"
        ),
    )

    parser.add_argument(
        "--prompt-only-output",
        default="data/generated/pseudo_paper_final_prompt_only.csv",
        help=(
            "Output path for the prompt-only dataset "
            "(default: data/generated/pseudo_paper_final_prompt_only.csv)"
        ),
    )

    parser.add_argument(
        "--check-model-ready",
        default=None,
        help=(
            "Optional historical model-ready CSV. "
            "Generated output must match it exactly."
        ),
    )

    parser.add_argument(
        "--check-prompt-only",
        default=None,
        help=(
            "Optional historical prompt-only CSV. "
            "Generated output must match it exactly."
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    model_ready_output = Path(args.model_ready_output)
    prompt_only_output = Path(args.prompt_only_output)

    source = validate_source(load_csv(input_path))
    model_ready = build_model_ready(source)
    validate_model_ready(model_ready)
    prompt_only = build_prompt_only(model_ready)

    verify_against(
        model_ready,
        Path(args.check_model_ready) if args.check_model_ready else None,
        "model-ready dataset",
    )

    verify_against(
        prompt_only,
        Path(args.check_prompt_only) if args.check_prompt_only else None,
        "prompt-only dataset",
    )

    write_csv(model_ready, model_ready_output)
    write_csv(prompt_only, prompt_only_output)

    print()
    print("Final dataset summary")
    print("=====================")
    print(f"Canonical source items : {len(source)}")
    print(f"Model-ready rows       : {len(model_ready)}")
    print(f"Prompt-only rows       : {len(prompt_only)}")
    print()
    print("Model-ready combination counts:")
    print(model_ready["comb"].value_counts().sort_index().to_string())
    print()
    print("Model-ready scope counts:")
    print(model_ready["gold_scope_label"].value_counts().to_string())
    print()
    print("Model-ready answer-position counts:")
    print(model_ready["gold_ans"].value_counts().sort_index().to_string())
    print()
    print("[OK] Final pseudosentence preprocessing completed successfully.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
