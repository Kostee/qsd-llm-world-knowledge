#!/usr/bin/env python3
"""
Unified zero-shot evaluation script for QSD.

Supports multiple providers via LangChain:
- OpenAI:       "gpt-4o", "gpt-4", "gpt-5.x", etc. (no prefix)
- Qwen:         "qwen:<model>" via DashScope OpenAI-compatible endpoint
- OpenRouter:   "openrouter:<model>" (e.g., Llama)
- Gemini:       "gemini:<model>"

Key features:
- Dataset selection: balanced / invented (CSV in data/private/)
- Example limit modes: 0 (no API calls), 5, all
- Majority vote over N repeats (default 5); tie -> random
- Writes per-run outputs under results/<run_id>/
- Skips work if outputs already exist
- RAG flag exists but is NOT implemented yet (placeholder)

Run from repository root.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import tenacity
from statistics import median
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError


# .env loading is optional but recommended (repo includes .env.example)
try:
    import dotenv

    dotenv.load_dotenv()
except Exception:
    pass

# LangChain components
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# -----------------------------
# Configuration / constants
# -----------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]  # .../qsd-llm-world-knowledge
DEFAULT_DATA_PRIVATE = REPO_ROOT / "data" / "private"

DEFAULT_BALANCED = DEFAULT_DATA_PRIVATE / "dataset_for_llms.csv"
DEFAULT_INVENTED = DEFAULT_DATA_PRIVATE / "invented_for_llms.csv"

DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

# Env vars expected (matching Maćkowy setup)
ENV_OPENAI = "OPENAI_API_KEY"
ENV_QWEN = "QWEN_API_KEY"
ENV_OPENROUTER = "OPENROUTER_API_KEY"
ENV_GOOGLE = "GOOGLE_API_KEY"


# -----------------------------
# Model Factory (from Maćkowy code, repo-ready)
# -----------------------------

class ModelFactory:
    """
    Factory for creating LLM instances across different providers.

    Prefixes:
    - (none)      -> OpenAI (e.g., "gpt-4o")
    - qwen:       -> Alibaba DashScope (e.g., "qwen:qwen-max")
    - openrouter: -> OpenRouter (e.g., "openrouter:meta-llama/llama-3.1-8b-instruct")
    - gemini:     -> Google AI (e.g., "gemini:gemini-1.5-pro")
    """

    @staticmethod
    def create(model_name: str, temperature: float = 0.0, max_tokens: int = 10):
        model_lower = model_name.lower()

        if model_lower.startswith("openrouter:"):
            return ModelFactory._create_openrouter(model_name, temperature, max_tokens)
        if model_lower.startswith("gemini:"):
            return ModelFactory._create_gemini(model_name, temperature, max_tokens)
        if model_lower.startswith("qwen:"):
            return ModelFactory._create_qwen(model_name, temperature, max_tokens)
        return ModelFactory._create_openai(model_name, temperature, max_tokens)

    @staticmethod
    def _create_openai(model_name: str, temperature: float, max_tokens: int):
        api_key = os.environ.get(ENV_OPENAI)
        if not api_key:
            raise RuntimeError(f"{ENV_OPENAI} is not set")
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

    @staticmethod
    def _create_qwen(model_name: str, temperature: float, max_tokens: int):
        api_key = os.environ.get(ENV_QWEN)
        if not api_key:
            raise RuntimeError(f"{ENV_QWEN} is not set")

        actual_model = model_name.split(":", 1)[1]
        kwargs: Dict[str, Any] = {
            "model": actual_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "api_key": api_key,
        }
        # Qwen3 can emit "thinking" by default; we disable it (Maćkowy trick)
        if "qwen3" in actual_model.lower():
            kwargs["extra_body"] = {"enable_thinking": False}
        return ChatOpenAI(**kwargs)

    @staticmethod
    def _create_openrouter(model_name: str, temperature: float, max_tokens: int):
        api_key = os.environ.get(ENV_OPENROUTER)
        if not api_key:
            raise RuntimeError(f"{ENV_OPENROUTER} is not set")

        actual_model = model_name.split(":", 1)[1]
        # Optional headers sometimes help OpenRouter; left minimal by default.
        return ChatOpenAI(
            model=actual_model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    @staticmethod
    def _create_gemini(model_name: str, temperature: float, max_tokens: int):
        google_key = os.environ.get(ENV_GOOGLE)
        if not google_key:
            raise RuntimeError(f"{ENV_GOOGLE} is not set")

        from langchain_google_genai import ChatGoogleGenerativeAI

        actual_model = model_name.split(":", 1)[1]
        return ChatGoogleGenerativeAI(
            model=actual_model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=google_key,
        )

    @staticmethod
    def safe_name(model_name: str) -> str:
        return model_name.replace("/", "_").replace(":", "_").replace(" ", "_")


# -----------------------------
# Data loading / normalization
# -----------------------------

REQUIRED_COLUMNS = ["sentence", "Option A", "Option B", "gold_ans"]

def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    # Basic sanity checks / normalization
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset missing required columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Normalize whitespace
    df["sentence"] = df["sentence"].astype(str).str.strip()
    df["Option A"] = df["Option A"].astype(str).str.strip()
    df["Option B"] = df["Option B"].astype(str).str.strip()
    df["gold_ans"] = df["gold_ans"].astype(str).str.strip()

    # gold_ans should be A/B
    df["gold_ans"] = df["gold_ans"].str.upper().str.replace(r"[^AB]", "", regex=True)

    return df


# -----------------------------
# Prompting / parsing / voting
# -----------------------------

PROMPT_TEMPLATE = (
    "{sentence}\n\n"
    "On the basis of this phrase/statement alone, and with no further context, "
    "there are two options:\n\n"
    "Option A   {opt_a}\n"
    "Option B   {opt_b}\n\n"
    "Which of these two options is most likely?"
)

SYSTEM_MESSAGE = "Reply only with 'Option A' or 'Option B'."

OPTION_A_PAT = re.compile(r"\b(option\s*a|^a\b)\.?", re.IGNORECASE)
OPTION_B_PAT = re.compile(r"\b(option\s*b|^b\b)\.?", re.IGNORECASE)

def parse_choice(text: str) -> Optional[str]:
    """
    Parse model output into 'A' or 'B'.
    More tolerant than strict equality, to reduce INVALIDs.
    """
    if not text:
        return None
    t = text.strip()

    # Common exact cases
    if t in ("Option A", "Option A.", "A", "A."):
        return "A"
    if t in ("Option B", "Option B.", "B", "B."):
        return "B"

    # Regex fallback
    if OPTION_A_PAT.search(t) and not OPTION_B_PAT.search(t):
        return "A"
    if OPTION_B_PAT.search(t) and not OPTION_A_PAT.search(t):
        return "B"

    # If both appear (rare), give up
    return None


@dataclass
class RunSpec:
    dataset_path: Path
    dataset_key: str           # "balanced" / "invented" / custom
    limit_mode: str            # "0" / "5" / "all"
    rag: bool
    model_name: str
    repeats: int
    seed: int


def majority_vote(choices: List[str], rng: random.Random) -> str:
    """
    Majority vote over choices (each 'A' or 'B').
    Tie-break: random.
    """
    count_a = sum(1 for c in choices if c == "A")
    count_b = sum(1 for c in choices if c == "B")
    if count_a > count_b:
        return "A"
    if count_b > count_a:
        return "B"
    # tie
    return rng.choice(["A", "B"])


def build_chain(model_name: str, temperature: float = 0.0, max_tokens: int = 64):
    """
    Create a prompt -> LLM -> parser chain.
    """
    llm = ModelFactory.create(model_name, temperature=temperature, max_tokens=max_tokens)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MESSAGE),
        ("user", "{prompt}"),
    ])
    return prompt | llm | StrOutputParser()


def rag_not_implemented() -> None:
    raise NotImplementedError(
        "RAG mode is not implemented yet. "
        "We will plug in dynamic retrieval (ConceptNet + Simple Wikipedia) later."
    )


# -----------------------------
# Metrics
# -----------------------------

def compute_group_accuracy(df: pd.DataFrame, group_col: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns dict: group_value -> {n, accuracy}
    """
    out: Dict[str, Dict[str, Any]] = {}
    for g, sub in df.groupby(group_col, dropna=False):
        n = int(len(sub))
        acc = float(sub["correct"].mean()) if n > 0 else 0.0
        out[str(g)] = {"n": n, "accuracy": acc}
    return out


def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    metrics["n"] = int(len(df))
    metrics["accuracy_overall"] = float(df["correct"].mean()) if len(df) else 0.0

    # By scope label, if available
    if "gold_scope_label" in df.columns:
        metrics["accuracy_by_gold_scope_label"] = compute_group_accuracy(df, "gold_scope_label")

    # By combination, if available
    if "comb" in df.columns:
        metrics["accuracy_by_comb"] = compute_group_accuracy(df, "comb")

    return metrics


def write_metrics(metrics: Dict[str, Any], out_dir: Path) -> None:
    # JSON
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Flat CSV summary for quick browsing
    rows = [
        {"metric": "accuracy_overall", "value": metrics.get("accuracy_overall"), "n": metrics.get("n")},
        {"metric": "runtime_seconds_total", "value": metrics.get("runtime_seconds_total"), "n": metrics.get("n")},
        {"metric": "runtime_mmss", "value": metrics.get("runtime_mmss"), "n": metrics.get("n")},
    ]

    if "accuracy_by_gold_scope_label" in metrics:
        for k, v in metrics["accuracy_by_gold_scope_label"].items():
            rows.append({"metric": f"accuracy_by_gold_scope_label::{k}", "value": v["accuracy"], "n": v["n"]})
    if "accuracy_by_comb" in metrics:
        for k, v in metrics["accuracy_by_comb"].items():
            rows.append({"metric": f"accuracy_by_comb::{k}", "value": v["accuracy"], "n": v["n"]})

    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)


# -----------------------------
# Main evaluation
# -----------------------------

def resolve_dataset(dataset_arg: str) -> Tuple[Path, str]:
    """
    Map dataset flag to actual file in data/private/.
    """
    dataset_arg = dataset_arg.lower().strip()
    if dataset_arg in ("balanced", "dataset_for_llms"):
        return DEFAULT_BALANCED, "balanced"
    if dataset_arg in ("invented", "invented_for_llms"):
        return DEFAULT_INVENTED, "invented"
    # Otherwise treat it as a path
    p = Path(dataset_arg)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p, p.stem


def resolve_limit_mode(limit_arg: str) -> str:
    """
    Accept: '0', '5', 'all'
    """
    limit_arg = str(limit_arg).lower().strip()
    if limit_arg not in ("0", "5", "all"):
        raise ValueError("--limit must be one of: 0, 5, all")
    return limit_arg


def make_run_id(spec: RunSpec) -> str:
    rag_key = "rag" if spec.rag else "no_rag"
    safe_model = ModelFactory.safe_name(spec.model_name)
    return f"{spec.dataset_key}__limit{spec.limit_mode}__{rag_key}__repeats{spec.repeats}__seed{spec.seed}__model{safe_model}"


def already_done(run_dir: Path) -> bool:
    return (run_dir / "predictions.csv").exists() and (run_dir / "metrics.json").exists()


def format_mmss(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def invoke_with_timeout(fn, timeout_s: int):
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        try:
            return fut.result(timeout=timeout_s)
        except FuturesTimeoutError:
            raise TimeoutError(f"LLM call timed out after {timeout_s}s")


def evaluate(spec: RunSpec) -> None:
    run_id = make_run_id(spec)
    out_dir = DEFAULT_RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config always (useful even if RAG is not implemented)
    config = {
        "dataset_path": str(spec.dataset_path),
        "dataset_key": spec.dataset_key,
        "limit_mode": spec.limit_mode,
        "rag": spec.rag,
        "model": spec.model_name,
        "repeats": spec.repeats,
        "seed": spec.seed,
        "timestamp_unix": int(time.time()),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if already_done(out_dir):
        print(f"[SKIP] already calculated: {out_dir}")
        return

    # RAG placeholder
    if spec.rag:
        # Create a friendly marker file then stop.
        (out_dir / "RAG_NOT_IMPLEMENTED.txt").write_text(
            "RAG mode requested but not implemented yet.\n", encoding="utf-8"
        )
        print(f"[ERROR] RAG requested but not implemented yet. See: {out_dir / 'RAG_NOT_IMPLEMENTED.txt'}")
        raise NotImplementedError("RAG mode is not implemented yet.")
    
    t_run_start = time.perf_counter()
    llm_call_durations: List[float] = []
    llm_call_failures = 0

    df = load_dataset(spec.dataset_path)

    # Limit selection
    if spec.limit_mode == "5":
        df_eval = df.head(5).copy()
    else:
        # '0' and 'all' both keep all rows
        df_eval = df.copy()

    # For limit=0: no API calls, but we still output as many rows as the dataset contains.
    do_api_calls = (spec.limit_mode != "0")

    # Build chain once per model
    chain = None
    if do_api_calls:
        chain = build_chain(spec.model_name, temperature=0.0, max_tokens=64)

    def _is_permanent_error(e: Exception) -> bool:
        msg = str(e).lower()
        return any(x in msg for x in [
            "401", "unauthorized",
            "403", "forbidden",
            "404", "not found",
            "model_not_found",
            "invalid api key",
        ])
    def _before_sleep(retry_state):
        exc = retry_state.outcome.exception()
        print(f"[RETRY] attempt={retry_state.attempt_number} after error: {exc}", flush=True)
    def _should_retry(exc: Exception) -> bool:
        return not _is_permanent_error(exc)
    
    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=2, min=2, max=60),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_exception(_should_retry),
        before_sleep=_before_sleep,
    )
    def call_once(prompt_text: str) -> str:
        assert chain is not None

        timeout_s = int(os.getenv("LLM_REQUEST_TIMEOUT_S", "90"))

        def _do_call():
            return chain.invoke({"prompt": prompt_text})

        # any exception here will be handled by tenacity (unless permanent)
        reply = invoke_with_timeout(_do_call, timeout_s=timeout_s)
        if reply is None:
            raise ValueError("Empty response: None")
        reply = str(reply).strip()
        if not reply:
            raise ValueError("Empty response: ''")

        if not isinstance(reply, str):
            reply = str(reply)

        parsed = parse_choice(reply)
        if parsed in ("A", "B"):
            return parsed

        # treat unparseable output as retryable (ValueError)
        raise ValueError(f"Unparseable response: {reply!r}")

    rows_out: List[Dict[str, Any]] = []

    for row_idx, (_, row) in enumerate(df_eval.iterrows()):
        sentence = row["sentence"]
        opt_a = row["Option A"]
        opt_b = row["Option B"]
        gold = row["gold_ans"]  # 'A' or 'B'

        prompt_text = PROMPT_TEMPLATE.format(sentence=sentence, opt_a=opt_a, opt_b=opt_b)

        per_run: List[str] = []
        if not do_api_calls:
            # Brutal "first-from-the-shelf" prediction: always Option A
            per_run = ["A"] * spec.repeats
        else:
            for r in range(spec.repeats):
                t0 = time.perf_counter()
                try:
                    choice = call_once(prompt_text)
                except Exception:
                    llm_call_failures += 1
                    choice = "A"
                finally:
                    llm_call_durations.append(time.perf_counter() - t0)

                per_run.append(choice)


        # Majority vote (tie -> random)
        # Use deterministic per-row RNG derived from global seed + row index
        row_rng = random.Random(spec.seed + row_idx * 10007)
        final_choice = majority_vote(per_run, row_rng)

        correct = (final_choice == gold)

        out = dict(row.to_dict())
        out["model"] = spec.model_name
        out["repeats"] = spec.repeats
        out["limit_mode"] = spec.limit_mode
        out["rag"] = spec.rag

        # Store raw votes + final
        for k, c in enumerate(per_run, start=1):
            out[f"pred_run_{k}"] = c
        out["prediction"] = final_choice
        out["correct"] = bool(correct)

        rows_out.append(out)
        if do_api_calls and (len(rows_out) % 25 == 0):
            elapsed = time.perf_counter() - t_run_start
            print(f"[PROGRESS] {len(rows_out)}/{len(df_eval)} done | elapsed {format_mmss(elapsed)}", flush=True)


    results_df = pd.DataFrame(rows_out)

    # Write predictions
    results_df.to_csv(out_dir / "predictions.csv", index=False)

    # Metrics
    metrics = compute_metrics(results_df)

    # --- runtime logging ---
    t_run_end = time.perf_counter()
    runtime_seconds = t_run_end - t_run_start
    metrics["runtime_seconds_total"] = float(runtime_seconds)
    metrics["runtime_mmss"] = format_mmss(runtime_seconds)

    if do_api_calls and llm_call_durations:
        d = sorted(llm_call_durations)
        metrics["llm_calls"] = int(len(d))
        metrics["llm_call_failures"] = int(llm_call_failures)
        metrics["llm_call_seconds_avg"] = float(sum(d) / len(d))
        metrics["llm_call_seconds_p50"] = float(median(d))
        # p90 (nearest-rank)
        p90_idx = max(0, min(len(d) - 1, int(0.90 * len(d)) - 1))
        metrics["llm_call_seconds_p90"] = float(d[p90_idx])

    write_metrics(metrics, out_dir)


    print(f"[DONE] Saved predictions to: {out_dir / 'predictions.csv'}")
    print(f"[DONE] Saved metrics to:      {out_dir / 'metrics.json'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QSD LLM zero-shot evaluation (baseline; RAG placeholder).")
    p.add_argument(
        "--dataset",
        type=str,
        default="balanced",
        help="Dataset selector: 'balanced' or 'invented', or a path. Default: balanced",
    )
    p.add_argument(
        "--limit",
        type=str,
        default="all",
        help="How many examples to process: 0 (no API calls; always Option A), 5 (first 5), all (full).",
    )
    p.add_argument(
        "--rag",
        action="store_true",
        help="Enable RAG mode (NOT IMPLEMENTED YET).",
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name. Examples: 'gpt-4o', 'qwen:qwen3-8b', 'openrouter:meta-llama/llama-3.1-8b-instruct', 'gemini:gemini-2.0-flash-exp'",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of calls per sentence (majority vote). Default: 5",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (tie-breaks). Default: 42",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path, dataset_key = resolve_dataset(args.dataset)
    limit_mode = resolve_limit_mode(args.limit)

    if args.repeats <= 0:
        raise ValueError("--repeats must be >= 1")

    spec = RunSpec(
        dataset_path=dataset_path,
        dataset_key=dataset_key,
        limit_mode=limit_mode,
        rag=bool(args.rag),
        model_name=args.model,
        repeats=int(args.repeats),
        seed=int(args.seed),
    )

    evaluate(spec)


if __name__ == "__main__":
    main()
