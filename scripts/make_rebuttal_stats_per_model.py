#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_rebuttal_stats_per_model.py

Generuje: results/rebuttal_bundle_extra.md

Co liczy:
- mean±SD po runach (zwykle 5) per model, per dataset (balanced/pseudosentences), per rag (True/False),
  oraz per scope_type (overall/surface/inverse).
- (E1) per-model różnicę balanced vs pseudosentences dla inverse:
  * unpaired cluster bootstrap (po itemach) 95% CI
  * 2-sample permutation test (po item-accuracies)
- (E2) per-model efekt RAG dla Combination IV:
  * paired bootstrap (po itemach) 95% CI
  * paired sign-flip permutation test (po item-diffs)
- dużo przykładów retrievalu dla Type IV (kandydaci, żeby wybrać 2–3 do rebuttalu)

Założenia:
- results/*/ ma co najmniej: config.json + predictions.csv
- predictions.csv ma 5 kolumn predykcji (np. model_pred_1..5 lub pred_run_1..5 itd.)
- gold jest w kolumnie typu: gold_ans / gold / gold_label / ...
- scope: preferred_scope / scope_type / ...
- combination: combination / comb / ...
- rag_context: rag_context / retrieved_context / ...

Uruchom:
  python scripts/make_rebuttal_stats_per_model.py
Opcje:
  --out results/rebuttal_bundle_extra.md
  --n_boot 10000
  --n_perm 20000
  --topk_examples 30
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ==============
# Konfiguracja
# ==============

# Jeśli chcesz ograniczyć do "paper models" (żeby nie mieszać modeli pobocznych),
# zostaw jak jest. Jeśli chcesz brać wszystko z results/, ustaw na None.
PAPER_MODELS_WHITELIST = {
    "GPT-5.1",
    "GPT-4o",
    "GPT-4o-mini",
    "Qwen-Max",
    "Qwen3-8B",
    "Llama-3.1-70B",
    "Llama-3.1-8B",
}

OUTLIER_MODEL = "Llama-3.1-8B"   # anomalny w pseudosentences (E1), ale w RAG (E2) zwykle nie wykluczamy


# ==============
# Utils: kolumny / normalizacja
# ==============

def _safe_load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _norm_colname(c: str) -> str:
    # lower + spacje/znaki -> _
    c = c.strip().lower()
    c = re.sub(r"[^\w]+", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c


def _normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Zwraca df z unifikowanymi nazwami kolumn + mapę {norm_name -> original_name}
    """
    mapping = {col: _norm_colname(col) for col in df.columns}
    inv = {mapping[k]: k for k in mapping}  # norm -> original
    df2 = df.rename(columns=mapping).copy()
    return df2, inv


def _col_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _detect_run_cols(df: pd.DataFrame) -> List[str]:
    """
    Szuka kolumn predykcji runów.
    Obsługuje m.in.:
      - pred_run_1..pred_run_5
      - model_pred_1..model_pred_5
      - run_1..run_5
      - pred_1..pred_5
    """
    patterns = [
        r"^pred_run_(\d+)$",
        r"^model_pred_(\d+)$",
        r"^run_(\d+)$",
        r"^pred_(\d+)$",
        r"^prediction_(\d+)$",
    ]

    found: List[Tuple[int, str]] = []
    for col in df.columns:
        for pat in patterns:
            m = re.match(pat, col)
            if m:
                found.append((int(m.group(1)), col))
                break

    if found:
        found.sort(key=lambda x: x[0])
        return [c for _, c in found]

    # fallback: single prediction column
    for c in ["prediction", "pred", "model_prediction", "model_pred"]:
        if c in df.columns:
            return [c]

    raise ValueError(f"Nie umiem wykryć kolumn runów w predictions.csv. Dostępne: {list(df.columns)}")


def _norm_ab(x) -> str:
    """Normalizuje różne kodowania na 'A' / 'B'."""
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip().lower()

    # typowe
    if s in {"a", "option a", "option_a", "opta", "answer a"}:
        return "A"
    if s in {"b", "option b", "option_b", "optb", "answer b"}:
        return "B"

    # 0/1 i 1/2
    if s in {"0", "1"}:
        return "A" if s == "0" else "B"
    if s == "2":
        return "B"

    # bool
    if s in {"true", "t", "yes"}:
        return "A"
    if s in {"false", "f", "no"}:
        return "B"

    # last resort: łap A/B w środku
    if re.search(r"\ba\b", s):
        return "A"
    if re.search(r"\bb\b", s):
        return "B"

    return s.upper()


def _norm_scope(x) -> str:
    if x is None:
        return "unknown"
    if isinstance(x, float) and np.isnan(x):
        return "unknown"
    s = str(x).strip().lower()
    if "inv" in s:
        return "inverse"
    if "surf" in s:
        return "surface"
    return s if s else "unknown"


def _norm_dataset_from_cfg_or_folder(cfg: dict, folder: str) -> str:
    # prefer config
    dk = str(cfg.get("dataset_key", cfg.get("dataset", cfg.get("dataset_name", "")))).lower()
    low = folder.lower()

    if "invented" in dk or "nonce" in dk or "pseudo" in dk or "invented" in low or "nonce" in low or "pseudo" in low:
        return "pseudosentences"
    if "balanced" in dk or "balanced" in low:
        return "balanced"
    return dk if dk else "unknown"


def _norm_rag_from_cfg_or_folder(cfg: dict, folder: str) -> bool:
    if isinstance(cfg.get("rag"), bool):
        return bool(cfg["rag"])
    low = folder.lower()
    if "no_rag" in low or "norag" in low:
        return False
    if "rag" in low:
        return True
    return False


def _canonical_model_from_cfg(cfg: dict, folder: str) -> str:
    raw = str(cfg.get("model", cfg.get("model_name", cfg.get("model_id", ""))))
    low = (raw or folder).lower()

    # usuń prefix dostawcy typu "qwen:" / "openrouter:"
    if ":" in low:
        low2 = low.split(":", 1)[1]
    else:
        low2 = low

    # mapowanie heurystyczne
    patterns = [
        (r"gpt[\-_]?5\.?1", "GPT-5.1"),
        (r"gpt[\-_]?4o[\-_]?mini", "GPT-4o-mini"),
        (r"gpt[\-_]?4o", "GPT-4o"),
        (r"qwen[\-_]?max", "Qwen-Max"),
        (r"qwen3[\-_]?8b", "Qwen3-8B"),
        (r"llama[\-_]?3\.?1[\-_]?70b", "Llama-3.1-70B"),
        (r"llama[\-_]?3\.?1[\-_]?8b", "Llama-3.1-8B"),
    ]
    for pat, canon in patterns:
        if re.search(pat, low2):
            return canon

    # fallback: oryginał
    return raw.strip() if raw.strip() else folder


def _to_long(pred: pd.DataFrame, run_cols: List[str]) -> pd.DataFrame:
    """
    pred ma już znormalizowane kolumny (lower + underscores).
    """
    gold_col = _col_first(pred, [
        "gold_ans", "gold", "gold_label", "gold_answer", "gold_option", "gold_choice",
        "label", "target", "answer", "correct_answer", "answer_key", "key"
    ])
    if gold_col is None:
        # heurystycznie
        heur = [c for c in pred.columns if re.search(r"(gold|label|target|correct|answer|key)", c)]
        heur = [c for c in heur if c not in run_cols]
        if len(heur) == 1:
            gold_col = heur[0]
    if gold_col is None:
        raise ValueError(f"Brak kolumny gold. Kolumny: {list(pred.columns)}")

    item_id_col = _col_first(pred, ["item_id", "id", "uid", "example_id"])
    if item_id_col is None:
        pred = pred.copy()
        pred["item_id"] = np.arange(len(pred))
        item_id_col = "item_id"

    scope_col = _col_first(pred, ["preferred_scope", "gold_scope_type", "scope_type", "scope", "reading"])
    comb_col = _col_first(pred, ["combination", "comb", "type", "qsd_type", "pattern_id"])

    sent_col = _col_first(pred, ["sentence", "text", "premise", "input_sentence"])
    a_col = _col_first(pred, ["option_a", "a", "opt_a"])
    b_col = _col_first(pred, ["option_b", "b", "opt_b"])

    rag_ctx_col = _col_first(pred, ["rag_context", "retrieved_context", "retrieval_context", "context", "docs"])
    src_col = _col_first(pred, ["source", "rag_source", "retrieval_source"])

    rows = []
    for rc in run_cols:
        m = re.search(r"(\d+)$", rc)
        run_idx = int(m.group(1)) if m else 1

        tmp = pred[[item_id_col, gold_col]].copy()
        tmp["run"] = run_idx
        tmp["pred_raw"] = pred[rc]
        tmp["gold_raw"] = pred[gold_col]

        tmp["pred"] = tmp["pred_raw"].apply(_norm_ab)
        tmp["gold"] = tmp["gold_raw"].apply(_norm_ab)
        tmp["correct"] = (tmp["pred"] == tmp["gold"]).astype(int)

        tmp["scope_type"] = pred[scope_col].apply(_norm_scope) if scope_col else "unknown"

        if comb_col:
            c = pred[comb_col]
            # próbuj na int 1..4
            def _to_int(v):
                if v is None:
                    return np.nan
                if isinstance(v, float) and np.isnan(v):
                    return np.nan
                s = str(v).strip().upper()
                roman = {"I": 1, "II": 2, "III": 3, "IV": 4}
                if s in roman:
                    return roman[s]
                try:
                    return int(float(s))
                except Exception:
                    return np.nan
            tmp["comb"] = c.apply(_to_int)
        else:
            tmp["comb"] = np.nan

        if sent_col: tmp["sentence"] = pred[sent_col].astype(str)
        else: tmp["sentence"] = ""

        if a_col: tmp["option_a"] = pred[a_col].astype(str)
        else: tmp["option_a"] = ""

        if b_col: tmp["option_b"] = pred[b_col].astype(str)
        else: tmp["option_b"] = ""

        if rag_ctx_col: tmp["rag_context"] = pred[rag_ctx_col].astype(str)
        else: tmp["rag_context"] = ""

        if src_col: tmp["source"] = pred[src_col].astype(str)
        else: tmp["source"] = ""

        rows.append(tmp)

    out = pd.concat(rows, ignore_index=True)
    out = out.rename(columns={item_id_col: "item_id"})
    return out


# ==============
# Statystyki
# ==============

def _run_mean_sd(df: pd.DataFrame) -> Tuple[float, float, int]:
    """
    mean i SD liczone po run-ach (run-level accuracy).
    Zwraca: (mean, sd, n_runs)
    """
    if df.empty:
        return (float("nan"), float("nan"), 0)
    by_run = df.groupby("run")["correct"].mean()
    mean = float(by_run.mean())
    sd = float(by_run.std(ddof=1)) if len(by_run) > 1 else 0.0
    return mean, sd, int(len(by_run))


def _item_acc(df: pd.DataFrame) -> pd.Series:
    """
    Item-level accuracy: średnia po runach (i ewentualnie powtórkach w obrębie tego samego runa).
    """
    if df.empty:
        return pd.Series(dtype=float)
    return df.groupby("item_id")["correct"].mean()


def _unpaired_delta_ci_perm(a: pd.Series, b: pd.Series, rng: np.random.Generator,
                           n_boot: int, n_perm: int) -> Tuple[float, Tuple[float, float], float]:
    """
    a,b: item-accuracies (Series/array) dla dwóch warunków (unpaired).
    delta = mean(b) - mean(a)
    CI: bootstrap po itemach w każdej grupie osobno
    p: permutation test (2-sample) na poziomie item-accuracies
    """
    a_vals = np.asarray(a.dropna().values, dtype=float)
    b_vals = np.asarray(b.dropna().values, dtype=float)

    if len(a_vals) == 0 or len(b_vals) == 0:
        return float("nan"), (float("nan"), float("nan")), float("nan")

    obs = float(b_vals.mean() - a_vals.mean())

    # bootstrap
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sa = rng.choice(a_vals, size=len(a_vals), replace=True)
        sb = rng.choice(b_vals, size=len(b_vals), replace=True)
        boots[i] = float(sb.mean() - sa.mean())
    lo, hi = np.quantile(boots, [0.025, 0.975])

    # permutation test
    comb = np.concatenate([a_vals, b_vals])
    n_a = len(a_vals)
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(comb)
        d = float(perm[n_a:].mean() - perm[:n_a].mean())
        if abs(d) >= abs(obs):
            cnt += 1
    p = (cnt + 1) / (n_perm + 1)

    return obs, (float(lo), float(hi)), float(p)


def _paired_delta_ci_signflip(diffs: np.ndarray, rng: np.random.Generator,
                             n_boot: int, n_perm: int) -> Tuple[float, Tuple[float, float], float]:
    """
    diffs: per-item różnice (rag - no_rag), np. z item-accuracies.
    delta = mean(diffs)
    CI: bootstrap po itemach (po diffs)
    p: paired sign-flip permutation test
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return float("nan"), (float("nan"), float("nan")), float("nan")

    obs = float(diffs.mean())

    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        s = rng.choice(diffs, size=len(diffs), replace=True)
        boots[i] = float(s.mean())
    lo, hi = np.quantile(boots, [0.025, 0.975])

    # sign-flip
    cnt = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(diffs), replace=True)
        d = float((diffs * signs).mean())
        if abs(d) >= abs(obs):
            cnt += 1
    p = (cnt + 1) / (n_perm + 1)

    return obs, (float(lo), float(hi)), float(p)


def _fmt(x, nd=3) -> str:
    if x is None:
        return "NA"
    try:
        if isinstance(x, float) and np.isnan(x):
            return "NA"
    except Exception:
        pass
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


# ==============
# Główne: load + compute + write
# ==============

def load_all_results(repo_root: Path, keep_only_paper_models: bool = True) -> pd.DataFrame:
    results_dir = repo_root / "results"
    if not results_dir.exists():
        raise RuntimeError(f"Brak katalogu results/ w repo: {results_dir}")

    all_parts = []
    for fd in sorted([p for p in results_dir.iterdir() if p.is_dir()]):
        cfg_p = fd / "config.json"
        pred_p = fd / "predictions.csv"
        if not cfg_p.exists() or not pred_p.exists():
            continue

        cfg = _safe_load_json(cfg_p)
        model = _canonical_model_from_cfg(cfg, fd.name)
        dataset = _norm_dataset_from_cfg_or_folder(cfg, fd.name)
        rag = _norm_rag_from_cfg_or_folder(cfg, fd.name)

        # filtr na paper models
        if keep_only_paper_models and PAPER_MODELS_WHITELIST and model not in PAPER_MODELS_WHITELIST:
            continue

        pred_raw = pd.read_csv(pred_p)
        pred, inv = _normalize_columns(pred_raw)
        run_cols = _detect_run_cols(pred)

        long = _to_long(pred, run_cols)
        long["model"] = model
        long["dataset"] = dataset
        long["rag"] = bool(rag)

        all_parts.append(long)

    if not all_parts:
        raise RuntimeError("Nie znalazłem żadnych pasujących folderów results/*/ z config.json + predictions.csv")

    df = pd.concat(all_parts, ignore_index=True)

    # sanity: comb jako int
    if "comb" in df.columns:
        df["comb"] = pd.to_numeric(df["comb"], errors="coerce")

    return df


def make_tables(df: pd.DataFrame, rng: np.random.Generator, n_boot: int, n_perm: int,
                topk_examples: int) -> str:
    lines: List[str] = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    models = sorted(df["model"].unique().tolist())
    lines.append("# Rebuttal bundle EXTRA (per-model stats)\n")
    lines.append(f"_Auto-generated on {ts}_\n")
    lines.append("This file complements `rebuttal_bundle.md` with **per-model** uncertainty/statistical tests and a larger pool of **RAG retrieval examples**.\n")

    # --------------------
    # 1) mean±SD po runach
    # --------------------
    lines.append("## A) Run-level variability (mean ± SD over runs)\n")
    lines.append("Run-level SD is computed from per-run accuracies (typically 5 runs). Below: overall / surface / inverse.\n")

    def add_run_table(dataset: str, rag: bool, title: str):
        sub = df[(df["dataset"] == dataset) & (df["rag"] == rag)]
        rows = []
        for m in models:
            dm = sub[sub["model"] == m]
            o_mean, o_sd, n_runs = _run_mean_sd(dm)
            s_mean, s_sd, _ = _run_mean_sd(dm[dm["scope_type"] == "surface"])
            i_mean, i_sd, _ = _run_mean_sd(dm[dm["scope_type"] == "inverse"])
            rows.append({
                "model": m,
                "n_items": int(dm["item_id"].nunique()),
                "n_runs": n_runs,
                "overall_mean": o_mean, "overall_sd": o_sd,
                "surface_mean": s_mean, "surface_sd": s_sd,
                "inverse_mean": i_mean, "inverse_sd": i_sd,
            })
        t = pd.DataFrame(rows)
        for c in ["overall_mean","overall_sd","surface_mean","surface_sd","inverse_mean","inverse_sd"]:
            t[c] = t[c].map(lambda x: _fmt(x, 3))
        lines.append(f"### {title}\n")
        lines.append(t.to_markdown(index=False))
        lines.append("")

    add_run_table("balanced", False, "Balanced (no-RAG)")
    add_run_table("pseudosentences", False, "Pseudosentences (no-RAG)")
    add_run_table("balanced", True, "Balanced (RAG)")

    # ------------------------------------------
    # 2) E1: balanced vs pseudosentences inverse
    # ------------------------------------------
    lines.append("\n## B) Per-model significance: balanced vs pseudosentences (inverse)\n")
    lines.append("We test the inverse-reading drop **per model** using item-level accuracies.\n")
    lines.append("- Delta is defined as: **Δ = acc(pseudosentences, inverse) − acc(balanced, inverse)**.\n")
    lines.append("- CI: unpaired cluster bootstrap over items (95%).\n")
    lines.append("- p-value: 2-sample permutation test over item-accuracies.\n")
    lines.append("\nWe report both **including** and **excluding** the pseudosentence outlier model (`Llama-3.1-8B`) to match reviewer concerns.\n")

    rows_inc = []
    rows_exc = []

    for m in models:
        bal_inv = df[(df.model == m) & (df.dataset == "balanced") & (~df.rag) & (df.scope_type == "inverse")]
        pse_inv = df[(df.model == m) & (df.dataset == "pseudosentences") & (~df.rag) & (df.scope_type == "inverse")]

        a = _item_acc(bal_inv)
        b = _item_acc(pse_inv)
        delta, ci, p = _unpaired_delta_ci_perm(a, b, rng, n_boot, n_perm)

        rows_inc.append({
            "model": m,
            "bal_inv_acc": float(a.mean()) if len(a) else float("nan"),
            "pse_inv_acc": float(b.mean()) if len(b) else float("nan"),
            "delta": delta,
            "ci_low": ci[0],
            "ci_high": ci[1],
            "p_perm": p,
            "n_bal_items": int(a.shape[0]),
            "n_pse_items": int(b.shape[0]),
        })

        if m == OUTLIER_MODEL:
            # w wersji "exclude" wpisz NA
            rows_exc.append({
                "model": m,
                "bal_inv_acc": float("nan"),
                "pse_inv_acc": float("nan"),
                "delta": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "p_perm": float("nan"),
                "n_bal_items": int(a.shape[0]),
                "n_pse_items": int(b.shape[0]),
            })
        else:
            rows_exc.append(rows_inc[-1].copy())

    def _rows_to_md(rows: List[dict], title: str):
        t = pd.DataFrame(rows)
        for c in ["bal_inv_acc", "pse_inv_acc", "delta", "ci_low", "ci_high"]:
            t[c] = t[c].map(lambda x: _fmt(x, 3))
        t["p_perm"] = t["p_perm"].map(lambda x: _fmt(x, 4))
        lines.append(f"### {title}\n")
        lines.append(t[[
            "model", "bal_inv_acc", "pse_inv_acc", "delta", "ci_low", "ci_high", "p_perm", "n_bal_items", "n_pse_items"
        ]].to_markdown(index=False))
        lines.append("")

    _rows_to_md(rows_inc, "E1 inverse drop (INCLUDING all models)")
    _rows_to_md(rows_exc, f"E1 inverse drop (EXCLUDING outlier {OUTLIER_MODEL})")

    # ------------------------------------------
    # 3) E2: RAG gain for Combination IV per model
    # ------------------------------------------
    lines.append("\n## C) Per-model significance: RAG effect on Combination IV (Type IV)\n")
    lines.append("We test the RAG gain **per model** for **Combination IV** using item-level paired differences.\n")
    lines.append("- Delta is: **Δ = acc(RAG) − acc(no-RAG)** (paired by item).\n")
    lines.append("- CI: paired bootstrap over items (95%).\n")
    lines.append("- p-value: paired sign-flip permutation test.\n")
    lines.append("\nNote: by default we **do not exclude** `Llama-3.1-8B` for RAG (per our earlier experimental protocol), but we report both variants for convenience.\n")

    def rag_effect_table(exclude_outlier: bool) -> pd.DataFrame:
        rows = []
        for m in models:
            if exclude_outlier and m == OUTLIER_MODEL:
                rows.append({
                    "model": m,
                    "no_rag_acc": float("nan"),
                    "rag_acc": float("nan"),
                    "delta": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "p_signflip": float("nan"),
                    "n_items_paired": 0,
                })
                continue

            no = df[(df.model == m) & (df.dataset == "balanced") & (~df.rag) & (df["comb"] == 4)]
            ra = df[(df.model == m) & (df.dataset == "balanced") & (df.rag) & (df["comb"] == 4)]

            a = _item_acc(no)
            b = _item_acc(ra)
            common = a.index.intersection(b.index)
            diffs = (b.loc[common] - a.loc[common]).to_numpy(dtype=float)

            delta, ci, p = _paired_delta_ci_signflip(diffs, rng, n_boot, n_perm)

            rows.append({
                "model": m,
                "no_rag_acc": float(a.loc[common].mean()) if len(common) else float("nan"),
                "rag_acc": float(b.loc[common].mean()) if len(common) else float("nan"),
                "delta": delta,
                "ci_low": ci[0],
                "ci_high": ci[1],
                "p_signflip": p,
                "n_items_paired": int(len(common)),
            })
        return pd.DataFrame(rows)

    t_inc = rag_effect_table(exclude_outlier=False)
    t_exc = rag_effect_table(exclude_outlier=True)

    def _fmt_rag_table(t: pd.DataFrame, title: str):
        tt = t.copy()
        for c in ["no_rag_acc", "rag_acc", "delta", "ci_low", "ci_high"]:
            tt[c] = tt[c].map(lambda x: _fmt(x, 3))
        tt["p_signflip"] = tt["p_signflip"].map(lambda x: _fmt(x, 4))
        lines.append(f"### {title}\n")
        lines.append(tt[[
            "model", "no_rag_acc", "rag_acc", "delta", "ci_low", "ci_high", "p_signflip", "n_items_paired"
        ]].to_markdown(index=False))
        lines.append("")

    _fmt_rag_table(t_inc, "E2 Type-IV RAG gain (INCLUDING all models)")
    _fmt_rag_table(t_exc, f"E2 Type-IV RAG gain (EXCLUDING {OUTLIER_MODEL})")

    # Bonus: RAG gain per combination (1..4)
    lines.append("\n### Bonus: RAG gain per model *and* per combination (1..4)\n")
    lines.append("This is useful if you want to check whether only Combination IV shows a consistent positive effect.\n")

    bonus_rows = []
    for m in models:
        for comb in [1, 2, 3, 4]:
            no = df[(df.model == m) & (df.dataset == "balanced") & (~df.rag) & (df["comb"] == comb)]
            ra = df[(df.model == m) & (df.dataset == "balanced") & (df.rag) & (df["comb"] == comb)]
            a = _item_acc(no)
            b = _item_acc(ra)
            common = a.index.intersection(b.index)
            diffs = (b.loc[common] - a.loc[common]).to_numpy(dtype=float)
            delta, ci, p = _paired_delta_ci_signflip(diffs, rng, n_boot, max(2000, n_perm // 5))  # lżej
            bonus_rows.append({
                "model": m,
                "comb": comb,
                "delta": delta,
                "ci_low": ci[0],
                "ci_high": ci[1],
                "p_signflip": p,
                "n_items": int(len(common)),
            })
    tb = pd.DataFrame(bonus_rows)
    for c in ["delta", "ci_low", "ci_high"]:
        tb[c] = tb[c].map(lambda x: _fmt(x, 3))
    tb["p_signflip"] = tb["p_signflip"].map(lambda x: _fmt(x, 4))
    lines.append(tb[["model", "comb", "delta", "ci_low", "ci_high", "p_signflip", "n_items"]].to_markdown(index=False))
    lines.append("")

    # ------------------------------------------
    # 4) Retrieval examples: dużo kandydatów
    # ------------------------------------------
    lines.append("\n## D) RAG retrieval examples: candidate pool (Type IV)\n")
    lines.append(f"Below is a larger pool (top {topk_examples}) of Type-IV items ranked by mean improvement **(RAG − no-RAG)** across models/runs.\n")
    lines.append("For each item we show 1–3 representative retrieved contexts (truncated) to help you pick 2–3 best-looking examples for the rebuttal.\n")
    lines.append("Note: retrieved content can be **ConceptNet triples** and/or **Simple Wikipedia sentences** (depending on what was returned for that item).\n")

    no4 = df[(df.dataset == "balanced") & (~df.rag) & (df["comb"] == 4)].copy()
    ra4 = df[(df.dataset == "balanced") & (df.rag) & (df["comb"] == 4)].copy()

    key = ["model", "item_id", "run"]
    merged = no4[key + ["correct"]].merge(
        ra4[key + ["correct", "sentence", "option_a", "option_b", "gold", "scope_type", "rag_context", "source"]],
        on=key,
        suffixes=("_no", "_rag"),
        how="inner"
    )
    if merged.empty:
        lines.append("\n⚠️ No merged no-RAG/RAG rows found for Combination IV. Check that both conditions exist and share item_id/run.\n")
        return "\n".join(lines)

    merged["diff"] = merged["correct_rag"] - merged["correct_no"]

    item_score = merged.groupby("item_id")["diff"].mean().sort_values(ascending=False)
    top_items = item_score.head(topk_examples).index.tolist()

    def _is_sentence_like(ctx: str) -> bool:
        if not ctx:
            return False
        ctx = str(ctx)
        # heurystyka: zdania częściej mają kropki i dłuższe fragmenty
        return (ctx.count(".") >= 1 and len(ctx.split()) >= 10)

    def _pick_contexts_for_item(item_df: pd.DataFrame, max_ctx: int = 3, truncate: int = 800) -> List[str]:
        # prefer: diff==1 oraz sentence-like (żeby złapać wiki)
        cands = item_df.copy()

        # sort preference
        cands["has_flip"] = (cands["diff"] == 1).astype(int)
        cands["sent_like"] = cands["rag_context"].apply(_is_sentence_like).astype(int)
        cands["ctx_len"] = cands["rag_context"].astype(str).str.len()

        cands = cands.sort_values(by=["has_flip", "sent_like", "ctx_len"], ascending=False)

        contexts = []
        seen = set()
        for _, r in cands.iterrows():
            ctx = str(r.get("rag_context", "") or "").strip()
            if not ctx:
                continue
            ctx_norm = re.sub(r"\s+", " ", ctx)
            if ctx_norm in seen:
                continue
            seen.add(ctx_norm)
            if len(ctx) > truncate:
                ctx = ctx[:truncate].rstrip() + " …"
            contexts.append(ctx)
            if len(contexts) >= max_ctx:
                break
        return contexts

    for iid in top_items:
        sub = merged[merged["item_id"] == iid].copy()
        if sub.empty:
            continue

        score = float(item_score.loc[iid])
        # reprezentatywne meta (weź pierwszy po preferencji flip + sentence-like)
        sub2 = sub.copy()
        sub2["has_flip"] = (sub2["diff"] == 1).astype(int)
        sub2["sent_like"] = sub2["rag_context"].apply(_is_sentence_like).astype(int)
        sub2 = sub2.sort_values(by=["has_flip", "sent_like"], ascending=False)
        rep = sub2.iloc[0].to_dict()

        sentence = (rep.get("sentence", "") or "").strip()
        opt_a = (rep.get("option_a", "") or "").strip()
        opt_b = (rep.get("option_b", "") or "").strip()
        scope = (rep.get("scope_type", "") or "").strip()

        lines.append(f"\n### Item {iid} (mean Δ={score:+.3f})\n")
        if sentence:
            lines.append(f"- **Sentence:** {sentence}\n")
        if opt_a:
            lines.append(f"- **Option A:** {opt_a}\n")
        if opt_b:
            lines.append(f"- **Option B:** {opt_b}\n")
        if scope:
            lines.append(f"- **Gold scope type:** {scope}\n")

        # pokaż kilka kontekstów
        contexts = _pick_contexts_for_item(sub, max_ctx=3, truncate=900)
        if not contexts:
            lines.append("- **Retrieved context:** (empty / not logged)\n")
        else:
            lines.append("- **Retrieved context (representative, truncated):**\n")
            for j, ctx in enumerate(contexts, 1):
                # markdown quote
                ctx_clean = ctx.replace("\n", " ").strip()
                lines.append(f"  - (ctx {j}) {ctx_clean}\n")

        # pokaż krótko ile flipów
        flips = int((sub["diff"] == 1).sum())
        total = int(len(sub))
        lines.append(f"- **Flip-to-correct count (rows):** {flips}/{total}\n")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/rebuttal_bundle_extra.md")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--n_perm", type=int, default=20000)
    ap.add_argument("--topk_examples", type=int, default=30)
    ap.add_argument("--include_nonpaper_models", action="store_true",
                    help="If set, do not filter to PAPER_MODELS_WHITELIST.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    df = load_all_results(repo_root, keep_only_paper_models=(not args.include_nonpaper_models))

    md = make_tables(
        df=df,
        rng=rng,
        n_boot=args.n_boot,
        n_perm=args.n_perm,
        topk_examples=args.topk_examples
    )

    out_path.write_text(md, encoding="utf-8")
    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()