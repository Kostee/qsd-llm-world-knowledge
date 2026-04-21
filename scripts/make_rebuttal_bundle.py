#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import json
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ========= USER TUNABLES =========
OUT_MD_NAME = "rebuttal_bundle.md"

# ile bootstrapów / permutacji (bezpieczne i szybkie na typowych rozmiarach)
N_BOOT = 10000
N_PERM = 20000
RNG = np.random.default_rng(0)

# ile przykładów do (4) wypisać (Type IV, balanced, RAG helps)
N_EXAMPLES_TYPE4 = 30

# ile różnych snippetów retrievalu maks per item (żeby nie robić ściany tekstu)
MAX_CTX_SNIPPETS_PER_ITEM = 8
CTX_SNIPPET_CHARS = 700

# =================================

PAPER_MODELS: Dict[str, str] = {
    # OpenAI
    "gpt-5.1": "GPT-5.1",
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o-mini",
    # Qwen
    "qwen-max": "Qwen-Max",
    "qwen3-8b": "Qwen3-8B",
    # Llama (różne identyfikatory spotykane w config/folder)
    "meta-llama/llama-3.1-70b-instruct": "Llama-3.1-70B",
    "meta-llama/llama-3.1-8b-instruct": "Llama-3.1-8B",
    "llama-3.1-70b": "Llama-3.1-70B",
    "llama-3.1-8b": "Llama-3.1-8B",
    "llama 3.1 70b": "Llama-3.1-70B",
    "llama 3.1 8b": "Llama-3.1-8B",
}

OUTLIER_MODEL_CANON = "Llama-3.1-8B"


# ---------- helpers ----------
def _safe_load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _col_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _detect_run_cols(df: pd.DataFrame) -> List[str]:
    # pred_run_1..pred_run_5
    cols = [c for c in df.columns if re.fullmatch(r"pred_run_\d+", c)]
    if cols:
        return sorted(cols, key=lambda x: int(x.split("_")[-1]))

    # run_1..run_5
    cols = [c for c in df.columns if re.fullmatch(r"run_\d+", c)]
    if cols:
        return sorted(cols, key=lambda x: int(x.split("_")[-1]))

    # single pred col
    for c in ["prediction", "pred", "model_pred", "predicted", "answer_pred"]:
        if c in df.columns:
            return [c]

    raise ValueError(f"Cannot detect run columns in predictions.csv. Columns={list(df.columns)}")


def _canonical_model(cfg: dict, folder_name: str) -> Optional[str]:
    # 1) z config
    for key in ["model", "model_name", "model_id", "provider_model", "llm", "engine"]:
        if key in cfg and isinstance(cfg[key], str):
            raw = cfg[key].strip()
            if raw in PAPER_MODELS:
                return PAPER_MODELS[raw]
            low = raw.lower()
            if low in PAPER_MODELS:
                return PAPER_MODELS[low]

    # 2) z folderu (substring match)
    low = folder_name.lower()
    for raw, canon in PAPER_MODELS.items():
        if raw in low:
            return canon

    return None


def _detect_dataset_and_rag(folder_name: str, cfg: dict) -> Tuple[str, bool]:
    low = folder_name.lower()

    # dataset
    if any(k in low for k in ["pseudo", "pseudos", "pseudosent", "nonce", "invented"]):
        dataset = "pseudosentences"
    elif "balanced" in low:
        dataset = "balanced"
    else:
        dataset = str(cfg.get("dataset", cfg.get("dataset_name", "unknown"))).lower()
        # normalize
        if "pseudo" in dataset or "nonce" in dataset or "invented" in dataset:
            dataset = "pseudosentences"
        elif "bal" in dataset:
            dataset = "balanced"

    # rag
    rag = False
    if "no_rag" in low or "norag" in low:
        rag = False
    elif "rag" in low:
        rag = True
    if isinstance(cfg.get("rag"), bool):
        rag = cfg["rag"]

    return dataset, rag


def _norm_ab(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip().lower()

    if s in {"a", "option a", "option_a", "opta", "answer a"}:
        return "A"
    if s in {"b", "option b", "option_b", "optb", "answer b"}:
        return "B"

    # sometimes 0/1 or 1/2
    if s in {"0", "1"}:
        return "A" if s == "0" else "B"
    if s == "2":
        return "B"

    # if it's literally 'A.' etc
    if s.startswith("a"):
        return "A"
    if s.startswith("b"):
        return "B"

    # last resort
    if re.search(r"\boption\s*a\b", s):
        return "A"
    if re.search(r"\boption\s*b\b", s):
        return "B"
    if re.search(r"\ba\b", s):
        return "A"
    if re.search(r"\bb\b", s):
        return "B"

    return s.upper()


def _norm_scope(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "unknown"
    s = str(x).strip().lower()
    if "inv" in s:
        return "inverse"
    if "surf" in s:
        return "surface"
    if s in {"s"}:
        return "surface"
    if s in {"i"}:
        return "inverse"
    return s


def _norm_comb(x) -> float:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return np.nan
    # numeric
    try:
        v = int(str(x).strip())
        return float(v)
    except Exception:
        pass
    # roman
    roman = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
    s = str(x).strip().lower()
    return float(roman.get(s, np.nan))


def _to_long(pred: pd.DataFrame, run_cols: List[str]) -> pd.DataFrame:
    # --- IMPORTANT: your real column names ---
    gold_col = _col_first(pred, [
        "gold_ans",  # <-- your format
        "gold", "gold_label", "gold_answer", "gold_option", "gold_choice",
        "label", "target", "answer", "correct_answer",
        "correct_option", "preferred_option", "answer_key", "key"
    ])
    if gold_col is None:
        # heuristic: columns that look gold-like but NOT scope-like
        heur = [c for c in pred.columns if re.search(r"(gold|label|target|correct|answer|key)", c, re.I)]
        heur = [c for c in heur if "scope" not in c.lower()]
        heur = [c for c in heur if not re.search(r"pred_run_\d+|^pred_|^run_\d+$", c)]
        if "gold_ans" in heur:
            gold_col = "gold_ans"
        elif len(heur) == 1:
            gold_col = heur[0]

    if gold_col is None:
        raise ValueError(f"No gold column found. Columns={list(pred.columns)}")

    item_id_col = _col_first(pred, ["item_id", "id", "uid", "example_id"])
    if item_id_col is None:
        pred = pred.copy()
        pred["item_id"] = np.arange(len(pred))
        item_id_col = "item_id"

    scope_col = _col_first(pred, [
        "gold_scope_label",  # <-- your format
        "gold_scope_type", "scope_type", "reading", "scope", "gold_scope"
    ])

    comb_col = _col_first(pred, ["comb", "combination", "type", "qsd_type", "pattern_id"])

    rag_ctx_col = _col_first(pred, ["rag_context", "retrieved_context", "context", "retrieval", "docs"])

    sent_col = _col_first(pred, ["sentence", "premise", "input_sentence", "text"])
    a_col = _col_first(pred, ["Option A", "option_a", "A", "opt_a"])
    b_col = _col_first(pred, ["Option B", "option_b", "B", "opt_b"])

    rows = []
    for rc in run_cols:
        run_idx = int(re.findall(r"\d+", rc)[-1]) if re.findall(r"\d+", rc) else 1
        tmp = pred[[item_id_col, gold_col]].copy()
        tmp["run"] = run_idx
        tmp["pred_raw"] = pred[rc]
        tmp["gold_raw"] = pred[gold_col]

        tmp["pred"] = tmp["pred_raw"].map(_norm_ab)
        tmp["gold"] = tmp["gold_raw"].map(_norm_ab)
        tmp["correct"] = (tmp["pred"] == tmp["gold"]).astype(int)

        if scope_col is not None:
            tmp["scope_type"] = pred[scope_col].map(_norm_scope)
        else:
            tmp["scope_type"] = "unknown"

        if comb_col is not None:
            tmp["comb"] = pred[comb_col].map(_norm_comb)
        else:
            tmp["comb"] = np.nan

        if rag_ctx_col is not None:
            tmp["rag_context"] = pred[rag_ctx_col].astype(str)
        else:
            tmp["rag_context"] = ""

        if sent_col is not None:
            tmp["sentence"] = pred[sent_col].astype(str)
        else:
            tmp["sentence"] = ""

        if a_col is not None:
            tmp["option_a"] = pred[a_col].astype(str)
        else:
            tmp["option_a"] = ""

        if b_col is not None:
            tmp["option_b"] = pred[b_col].astype(str)
        else:
            tmp["option_b"] = ""

        rows.append(tmp)

    long = pd.concat(rows, ignore_index=True)
    long = long.rename(columns={item_id_col: "item_id"})
    return long


def _acc(df: pd.DataFrame) -> float:
    return float(df["correct"].mean()) if len(df) else float("nan")


def _run_stats(df: pd.DataFrame) -> Tuple[float, float]:
    by_run = df.groupby("run")["correct"].mean()
    mean = float(by_run.mean()) if len(by_run) else float("nan")
    sd = float(by_run.std(ddof=1)) if len(by_run) > 1 else 0.0
    return mean, sd


def _fmt(x: float) -> str:
    if x != x:
        return "NA"
    return f"{x:.3f}"


def _bootstrap_unpaired_delta(df_a: pd.DataFrame, df_b: pd.DataFrame, cluster="item_id", n_boot=N_BOOT) -> Tuple[float, Tuple[float, float]]:
    """
    Unpaired cluster bootstrap by item_id.
    Returns point estimate (b - a) and 95% CI.
    """
    a_items = df_a[cluster].dropna().unique()
    b_items = df_b[cluster].dropna().unique()

    point = float(df_b["correct"].mean() - df_a["correct"].mean())

    boots = []
    for _ in range(n_boot):
        sa = RNG.choice(a_items, size=len(a_items), replace=True)
        sb = RNG.choice(b_items, size=len(b_items), replace=True)
        ma = df_a[df_a[cluster].isin(sa)]["correct"].mean()
        mb = df_b[df_b[cluster].isin(sb)]["correct"].mean()
        boots.append(float(mb - ma))

    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, (float(lo), float(hi))


def _paired_item_diff(df_no: pd.DataFrame, df_rag: pd.DataFrame, keys=("model", "item_id", "run")) -> pd.DataFrame:
    """
    Pair noRAG vs RAG on (model,item_id,run).
    Returns df with columns keys + correct_no/correct_rag + diff.
    """
    key = list(keys)
    m = df_no[key + ["correct", "sentence", "option_a", "option_b", "gold", "rag_context"]].merge(
        df_rag[key + ["correct", "sentence", "option_a", "option_b", "gold", "rag_context"]],
        on=key,
        suffixes=("_no", "_rag"),
        how="inner",
    )
    m["diff"] = m["correct_rag"] - m["correct_no"]
    return m


def _bootstrap_paired_delta(paired: pd.DataFrame, cluster="item_id", n_boot=N_BOOT) -> Tuple[float, Tuple[float, float]]:
    diffs = paired.groupby(cluster)["diff"].mean().to_numpy()
    point = float(np.mean(diffs)) if len(diffs) else float("nan")
    boots = []
    for _ in range(n_boot):
        sample = RNG.choice(diffs, size=len(diffs), replace=True)
        boots.append(float(np.mean(sample)))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, (float(lo), float(hi))


def _signflip_pvalue(paired: pd.DataFrame, cluster="item_id", n_perm=N_PERM) -> float:
    diffs = paired.groupby(cluster)["diff"].mean().to_numpy()
    obs = float(np.mean(diffs)) if len(diffs) else 0.0
    cnt = 0
    for _ in range(n_perm):
        signs = RNG.choice([-1, 1], size=len(diffs), replace=True)
        m = float(np.mean(diffs * signs))
        if abs(m) >= abs(obs):
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def _collect_context_snippets(texts: List[str], max_snips=MAX_CTX_SNIPPETS_PER_ITEM, chars=CTX_SNIPPET_CHARS) -> List[str]:
    cleaned = []
    for t in texts:
        if not t:
            continue
        s = str(t).strip()
        if not s or s.lower() == "nan":
            continue
        s = re.sub(r"\s+", " ", s)
        if len(s) > chars:
            s = s[:chars].rstrip() + "…"
        cleaned.append(s)

    # unique, keep order
    seen = set()
    uniq = []
    for s in cleaned:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
        if len(uniq) >= max_snips:
            break
    return uniq


# ---------- main ----------
def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    out_path = results_dir / OUT_MD_NAME

    folders = [p for p in results_dir.iterdir() if p.is_dir()]

    all_records = []
    used_folders = []

    for fd in folders:
        cfg_p = fd / "config.json"
        pred_p = fd / "predictions.csv"
        if not cfg_p.exists() or not pred_p.exists():
            continue

        cfg = _safe_load_json(cfg_p)
        canon_model = _canonical_model(cfg, fd.name)
        if canon_model is None:
            continue

        dataset, rag = _detect_dataset_and_rag(fd.name, cfg)
        if dataset not in {"balanced", "pseudosentences"}:
            continue  # skupiamy się na tym co potrzebne do rebuttalu

        pred = pd.read_csv(pred_p)

        run_cols = _detect_run_cols(pred)
        long = _to_long(pred, run_cols)

        long["model"] = canon_model
        long["dataset"] = dataset
        long["rag"] = bool(rag)
        long["folder"] = fd.name

        all_records.append(long)
        used_folders.append(fd.name)

    if not all_records:
        raise RuntimeError("No matching paper-model folders found in results/")

    all_long = pd.concat(all_records, ignore_index=True)

    # sanity: czy scope rozpoznany?
    if (all_long["scope_type"] == "unknown").all():
        raise RuntimeError(
            "scope_type is unknown for all rows. Check that predictions.csv contains gold_scope_label "
            "or add correct column mapping in _to_long()."
        )

    # ---------- per-model summaries ----------
    def summarize(dataset: str, rag: bool, exclude_outlier: bool = False) -> pd.DataFrame:
        df = all_long[(all_long["dataset"] == dataset) & (all_long["rag"] == rag)].copy()
        if exclude_outlier:
            df = df[df["model"] != OUTLIER_MODEL_CANON]

        rows = []
        for m in sorted(df["model"].unique()):
            d = df[df["model"] == m]
            overall_mean, overall_sd = _run_stats(d)
            surf_mean, surf_sd = _run_stats(d[d["scope_type"] == "surface"])
            inv_mean, inv_sd = _run_stats(d[d["scope_type"] == "inverse"])
            rows.append({
                "model": m,
                "overall_mean": overall_mean, "overall_sd": overall_sd,
                "surface_mean": surf_mean, "surface_sd": surf_sd,
                "inverse_mean": inv_mean, "inverse_sd": inv_sd,
                "n_items": int(d["item_id"].nunique()),
                "n_runs": int(d["run"].nunique()),
            })
        return pd.DataFrame(rows)

    bal_simple = summarize("balanced", rag=False, exclude_outlier=False)
    bal_rag = summarize("balanced", rag=True, exclude_outlier=False)
    pse_simple = summarize("pseudosentences", rag=False, exclude_outlier=False)

    # ---------- (2) key statistics ----------
    # inverse drop: balanced -> pseudo (inverse only), exclude outlier
    df_bal_inv = all_long[
        (all_long["dataset"] == "balanced") &
        (all_long["rag"] == False) &
        (all_long["scope_type"] == "inverse") &
        (all_long["model"] != OUTLIER_MODEL_CANON)
    ].copy()

    df_pse_inv = all_long[
        (all_long["dataset"] == "pseudosentences") &
        (all_long["rag"] == False) &
        (all_long["scope_type"] == "inverse") &
        (all_long["model"] != OUTLIER_MODEL_CANON)
    ].copy()

    inv_bal_acc = _acc(df_bal_inv)
    inv_pse_acc = _acc(df_pse_inv)
    inv_delta_point, inv_delta_ci = _bootstrap_unpaired_delta(df_bal_inv, df_pse_inv, cluster="item_id", n_boot=N_BOOT)

    # human vs LLM asymmetry: (surface-inverse gap) humans vs LLMs
    # NOTE: humans are not in results/ so we cannot compute it here unless you store human preds.
    # We therefore just compute LLM surface-inverse gap per condition (balanced vs pseudo) excluding outlier.
    df_bal_llm = all_long[
        (all_long["dataset"] == "balanced") &
        (all_long["rag"] == False) &
        (all_long["model"] != OUTLIER_MODEL_CANON)
    ]
    df_pse_llm = all_long[
        (all_long["dataset"] == "pseudosentences") &
        (all_long["rag"] == False) &
        (all_long["model"] != OUTLIER_MODEL_CANON)
    ]
    bal_surface = _acc(df_bal_llm[df_bal_llm["scope_type"] == "surface"])
    bal_inverse = _acc(df_bal_llm[df_bal_llm["scope_type"] == "inverse"])
    pse_surface = _acc(df_pse_llm[df_pse_llm["scope_type"] == "surface"])
    pse_inverse = _acc(df_pse_llm[df_pse_llm["scope_type"] == "inverse"])
    gap_bal = bal_surface - bal_inverse
    gap_pse = pse_surface - pse_inverse

    # Type IV RAG gain (paired) — report both incl. outlier and excl. outlier
    def type4_stats(exclude_outlier: bool) -> Dict[str, object]:
        df_no = all_long[
            (all_long["dataset"] == "balanced") &
            (all_long["rag"] == False) &
            (all_long["comb"] == 4)
        ]
        df_ra = all_long[
            (all_long["dataset"] == "balanced") &
            (all_long["rag"] == True) &
            (all_long["comb"] == 4)
        ]
        if exclude_outlier:
            df_no = df_no[df_no["model"] != OUTLIER_MODEL_CANON]
            df_ra = df_ra[df_ra["model"] != OUTLIER_MODEL_CANON]

        paired = _paired_item_diff(df_no, df_ra, keys=("model", "item_id", "run"))
        delta, ci = _bootstrap_paired_delta(paired, cluster="item_id", n_boot=N_BOOT)
        p = _signflip_pvalue(paired, cluster="item_id", n_perm=N_PERM)

        # item ranking by mean improvement
        item_scores = paired.groupby("item_id")["diff"].mean().sort_values(ascending=False)

        return {
            "paired": paired,
            "delta": delta,
            "ci": ci,
            "p": p,
            "item_scores": item_scores,
        }

    t4_incl = type4_stats(exclude_outlier=False)
    t4_excl = type4_stats(exclude_outlier=True)

    # ---------- (4) many examples: Type IV where RAG helps ----------
    # We'll use the EXCLUDING outlier version for narrative stability.
    paired = t4_excl["paired"]
    item_scores = t4_excl["item_scores"]

    top_items = item_scores[item_scores > 0].head(N_EXAMPLES_TYPE4).index.tolist()

    examples = []
    for iid in top_items:
        sub = paired[paired["item_id"] == iid].copy()
        if len(sub) == 0:
            continue

        # aggregate correctness by condition
        no_acc = float(sub["correct_no"].mean())
        rag_acc = float(sub["correct_rag"].mean())
        n_pairs = int(len(sub))

        # pick a representative row for the sentence/options/gold
        # (prefer a row where it flips 0->1)
        flip = sub[(sub["correct_no"] == 0) & (sub["correct_rag"] == 1)]
        rep = (flip.iloc[0] if len(flip) else sub.iloc[0])

        sentence = rep.get("sentence_rag", "") or rep.get("sentence_no", "")
        opt_a = rep.get("option_a_rag", "") or rep.get("option_a_no", "")
        opt_b = rep.get("option_b_rag", "") or rep.get("option_b_no", "")
        gold = rep.get("gold_rag", "") or rep.get("gold_no", "")

        # collect multiple retrieval contexts from different runs/models
        ctxs = sub["rag_context_rag"].tolist()
        ctx_snips = _collect_context_snippets(ctxs, max_snips=MAX_CTX_SNIPPETS_PER_ITEM, chars=CTX_SNIPPET_CHARS)

        examples.append({
            "item_id": iid,
            "mean_diff": float(sub["diff"].mean()),
            "no_acc": no_acc,
            "rag_acc": rag_acc,
            "n_pairs": n_pairs,
            "sentence": sentence,
            "option_a": opt_a,
            "option_b": opt_b,
            "gold": gold,
            "ctx_snips": ctx_snips,
        })

    # ---------- write markdown ----------
    lines: List[str] = []
    lines.append("# Rebuttal bundle (auto-generated)\n")

    lines.append("## Sanity checks\n")
    lines.append(f"- Folders used: {len(used_folders)}\n")
    lines.append(f"- Models detected: {', '.join(sorted(all_long['model'].unique()))}\n")
    lines.append(f"- Datasets detected: {', '.join(sorted(all_long['dataset'].unique()))}\n")
    lines.append(f"- Runs detected (global): {int(all_long['run'].nunique())}\n")
    lines.append(f"- Outlier model (excluded in some aggregates): **{OUTLIER_MODEL_CANON}**\n")
    lines.append(f"- NOTE: scope_type distribution: {all_long['scope_type'].value_counts().to_dict()}\n")

    def _df_to_md(df: pd.DataFrame, title: str) -> None:
        lines.append(f"\n### {title}\n")
        view = df.copy()
        for c in ["overall_mean", "overall_sd", "surface_mean", "surface_sd", "inverse_mean", "inverse_sd"]:
            if c in view.columns:
                view[c] = view[c].map(_fmt)
        lines.append(view.to_markdown(index=False))

    lines.append("\n## Per-model results (mean ± SD over runs)\n")
    _df_to_md(bal_simple, "Balanced, no-RAG")
    _df_to_md(bal_rag, "Balanced, RAG")
    _df_to_md(pse_simple, "Pseudosentences, no-RAG")

    lines.append("\n## (2) Statistical reliability: key numbers for rebuttal\n")

    lines.append("\n### Inverse drop (balanced → pseudosentences), excluding outlier\n")
    lines.append(f"- balanced inverse accuracy = {_fmt(inv_bal_acc)}\n")
    lines.append(f"- pseudo inverse accuracy   = {_fmt(inv_pse_acc)}\n")
    lines.append(f"- Δ (pseudo - balanced) = {_fmt(inv_delta_point)} "
                 f"(95% cluster bootstrap CI [{_fmt(inv_delta_ci[0])}, {_fmt(inv_delta_ci[1])}])\n")

    lines.append("\n### LLM surface–inverse gap (excl. outlier; helpful for 'asymmetry' discussion)\n")
    lines.append(f"- balanced: surface={_fmt(bal_surface)}, inverse={_fmt(bal_inverse)}, gap(surface-inverse)={_fmt(gap_bal)}\n")
    lines.append(f"- pseudo:   surface={_fmt(pse_surface)}, inverse={_fmt(pse_inverse)}, gap(surface-inverse)={_fmt(gap_pse)}\n")

    lines.append("\n### Type IV (comb=4) RAG gain (paired by item), report both versions\n")
    lines.append(f"- EXCL outlier: Δ (RAG-noRAG) = {_fmt(t4_excl['delta'])} "
                 f"(95% paired bootstrap CI [{_fmt(t4_excl['ci'][0])}, {_fmt(t4_excl['ci'][1])}]); "
                 f"sign-flip p={t4_excl['p']:.4g}\n")
    lines.append(f"- INCL outlier: Δ (RAG-noRAG) = {_fmt(t4_incl['delta'])} "
                 f"(95% paired bootstrap CI [{_fmt(t4_incl['ci'][0])}, {_fmt(t4_incl['ci'][1])}]); "
                 f"sign-flip p={t4_incl['p']:.4g}\n")

    lines.append("\n## (4) Many Type IV examples where RAG helps (for concrete rebuttal evidence)\n")
    if not examples:
        lines.append("- No examples extracted. This usually means missing comb==4 rows under both rag/no_rag, "
                     "or missing rag_context column.\n")
    else:
        lines.append(f"- Extracted **{len(examples)}** items (top by mean improvement, only items with mean diff > 0, excl. outlier).\n")
        for ex in examples:
            lines.append(f"\n### Item {int(ex['item_id'])}  |  mean Δ={ex['mean_diff']:.3f}  |  noRAG={ex['no_acc']:.3f}  RAG={ex['rag_acc']:.3f}  |  pairs={ex['n_pairs']}\n")
            if ex["sentence"]:
                lines.append(f"- Sentence: {ex['sentence']}\n")
            if ex["option_a"]:
                lines.append(f"- Option A: {ex['option_a']}\n")
            if ex["option_b"]:
                lines.append(f"- Option B: {ex['option_b']}\n")
            if ex["gold"]:
                lines.append(f"- Gold: {ex['gold']}\n")
            if ex["ctx_snips"]:
                lines.append("- Retrieved context snippets (deduplicated, truncated):\n")
                for i, sn in enumerate(ex["ctx_snips"], start=1):
                    lines.append(f"  {i}. {sn}\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()
