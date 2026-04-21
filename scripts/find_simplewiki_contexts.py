#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scan all results/**/predictions.csv and extract rows whose retrieved context
looks like it comes from Simple Wikipedia (and not ConceptNet triples).

Outputs:
- results/simplewiki_hits.csv
- results/simplewiki_hits.md
"""

from __future__ import annotations
from pathlib import Path
import json
import re
import pandas as pd
from typing import Dict, List, Tuple, Optional

# ---------------------------
# Heuristics / patterns
# ---------------------------

# strong indicators of ConceptNet-style triple dumps
CONCEPTNET_PATTERNS = [
    r"—\s*[a-z_]+\s*→",          # e.g., "—partof→"
    r"\b(isa|partof|atlocation|capableof|usedfor|hasproperty|madeof|createdby)\b",
    r"/r/[A-Za-z_]+",            # ConceptNet relation paths sometimes
    r"\bconceptnet\b",
]

# strong indicators of (Simple) Wikipedia / wiki text
SIMPLEWIKI_PATTERNS = [
    r"\bsimple wikipedia\b",
    r"simple\.wikipedia\.org",
    r"\bwikipedia\b",
    r"\bIn\s+\d{3,4}\b",          # wiki-ish lead sentences sometimes contain dates; weak but useful with others
    r"\bis a\b.*\bthat\b",        # very weak; only used together with others
    r"\bwas born\b",
    r"\bis an?\b",
]

# columns that often contain retrieved context
CTX_COL_CANDIDATES = [
    "rag_context", "retrieved_context", "context", "retrieval", "docs",
    "ctx", "contexts", "passages", "passage", "retrieved_passages",
    "topk", "top_k", "evidence",
]

# columns that sometimes store per-doc info
SOURCE_COL_HINTS = ["source", "corpus", "dataset", "collection", "kb", "index"]
DOC_COL_HINTS = ["doc", "passage", "ctx", "context", "evidence", "retrieved", "chunk", "text"]


def safe_load_json(p: Path) -> Dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def find_context_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    lower_map = {c.lower(): c for c in df.columns}

    # direct hits by name
    for cand in CTX_COL_CANDIDATES:
        if cand in lower_map:
            cols.append(lower_map[cand])

    # heuristic: any col containing "rag" and "ctx/context/docs"
    for c in df.columns:
        lc = c.lower()
        if ("rag" in lc or "retriev" in lc) and ("ctx" in lc or "context" in lc or "doc" in lc or "passage" in lc):
            if c not in cols:
                cols.append(c)

    # heuristic: any col that contains "wiki" or "conceptnet" might be explicit source/text
    for c in df.columns:
        lc = c.lower()
        if "wiki" in lc or "conceptnet" in lc:
            if c not in cols:
                cols.append(c)

    return cols


def find_source_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        lc = c.lower()
        if any(h in lc for h in SOURCE_COL_HINTS) and any(h2 in lc for h2 in ["rag", "retriev", "ctx", "doc", "passage", "kb", "index", "corpus"]):
            cols.append(c)
    return cols


def looks_like_conceptnet(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    return any(re.search(pat, t, flags=re.IGNORECASE) for pat in CONCEPTNET_PATTERNS)


def looks_like_simplewiki(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    # at least one strong-ish wiki indicator
    return any(re.search(pat, t, flags=re.IGNORECASE) for pat in SIMPLEWIKI_PATTERNS)


def row_has_simplewiki(df_row: pd.Series, ctx_cols: List[str], src_cols: List[str]) -> Tuple[bool, str]:
    """
    Return (is_simplewiki, evidence_string).
    evidence_string is a short justification / matched snippet.
    """

    # 1) If there are explicit source columns and they mention simplewiki/wiki
    for c in src_cols:
        v = df_row.get(c, "")
        s = "" if pd.isna(v) else str(v)
        if re.search(r"(simple\s*wiki|simplewikipedia|simple\.wikipedia|wikipedia)", s, flags=re.IGNORECASE):
            # double-check: if it also screams conceptnet, treat as ambiguous but still keep
            return True, f"source_col={c}: {s[:200]}"

    # 2) Look inside candidate context columns
    for c in ctx_cols:
        v = df_row.get(c, "")
        s = "" if pd.isna(v) else str(v)

        if not s.strip():
            continue

        # If it looks like pure ConceptNet triples and no wiki hints, skip
        is_cn = looks_like_conceptnet(s)
        is_wk = looks_like_simplewiki(s)

        if is_wk and not is_cn:
            return True, f"context_col={c}: {s[:220].replace('\\n',' ')}"
        # If both match, still accept (might be mixed retrieval); mark ambiguous
        if is_wk and is_cn:
            return True, f"context_col={c} (MIXED): {s[:220].replace('\\n',' ')}"

    # 3) As a last resort, scan ALL columns for a URL hint (rare but useful)
    for c in df_row.index:
        v = df_row.get(c, "")
        s = "" if pd.isna(v) else str(v)
        if "simple.wikipedia.org" in s.lower():
            return True, f"any_col={c}: {s[:220].replace('\\n',' ')}"

    return False, ""


def detect_meta(folder: Path) -> Dict[str, str]:
    """
    Try to grab model/dataset/rag from config.json and folder name for nicer reporting.
    """
    cfg_path = folder / "config.json"
    cfg = safe_load_json(cfg_path) if cfg_path.exists() else {}
    name = folder.name.lower()

    model = str(cfg.get("model", cfg.get("model_name", cfg.get("model_id", "")))) or folder.name
    dataset = str(cfg.get("dataset", cfg.get("dataset_name", ""))) or ""
    rag = cfg.get("rag", None)

    # folder-name fallbacks
    if not dataset:
        if "invented" in name or "pseudos" in name or "nonce" in name:
            dataset = "pseudosentences"
        elif "balanced" in name:
            dataset = "balanced"
        else:
            dataset = "unknown"

    if rag is None:
        if "no_rag" in name or "norag" in name:
            rag = False
        elif "rag" in name:
            rag = True

    return {
        "folder": folder.name,
        "model_raw": model,
        "dataset": str(dataset),
        "rag": str(bool(rag)) if rag is not None else "unknown",
    }


def main():
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    out_csv = results_dir / "simplewiki_hits.csv"
    out_md = results_dir / "simplewiki_hits.md"

    if not results_dir.exists():
        raise SystemExit(f"[ERR] Missing results dir: {results_dir}")

    pred_paths = sorted(results_dir.glob("**/predictions.csv"))
    if not pred_paths:
        raise SystemExit("[ERR] No predictions.csv found under results/**/")

    hits = []
    scanned_files = 0
    scanned_rows = 0

    for pred_path in pred_paths:
        folder = pred_path.parent
        scanned_files += 1

        try:
            df = pd.read_csv(pred_path)
        except Exception as e:
            print(f"[WARN] Could not read {pred_path}: {e}")
            continue

        scanned_rows += len(df)

        ctx_cols = find_context_columns(df)
        src_cols = find_source_columns(df)

        # if no obvious context cols, still allow scanning all cols via row_has_simplewiki fallback
        meta = detect_meta(folder)

        for idx, row in df.iterrows():
            ok, evidence = row_has_simplewiki(row, ctx_cols, src_cols)
            if not ok:
                continue

            # attach a short snippet of the "best" context we can find
            snippet = ""
            for c in ctx_cols:
                v = row.get(c, "")
                s = "" if pd.isna(v) else str(v)
                if looks_like_simplewiki(s):
                    snippet = s
                    break
            if not snippet:
                # fallback: first non-empty ctx col
                for c in ctx_cols:
                    v = row.get(c, "")
                    s = "" if pd.isna(v) else str(v)
                    if s.strip():
                        snippet = s
                        break

            # Try to preserve key fields if present
            item_id = row.get("item_id", row.get("id", row.get("uid", "")))
            comb = row.get("comb", row.get("combination", row.get("type", "")))
            scope = row.get("gold_scope_type", row.get("scope_type", row.get("reading", "")))

            sent = row.get("sentence", row.get("premise", row.get("input_sentence", row.get("text", ""))))
            opt_a = row.get("option_a", row.get("A", row.get("opt_a", "")))
            opt_b = row.get("option_b", row.get("B", row.get("opt_b", "")))

            hits.append({
                **meta,
                "predictions_path": str(pred_path.relative_to(repo_root)),
                "row_index": int(idx),
                "item_id": item_id,
                "comb": comb,
                "scope_type": scope,
                "sentence": sent,
                "option_a": opt_a,
                "option_b": opt_b,
                "evidence": evidence,
                "context_snippet": (snippet[:1200] if isinstance(snippet, str) else ""),
            })

    hits_df = pd.DataFrame(hits)
    hits_df.to_csv(out_csv, index=False, encoding="utf-8")

    # --- markdown report ---
    lines = []
    lines.append("# Simple Wikipedia retrieval hits (auto-scan)\n")
    lines.append(f"- Scanned files: **{scanned_files}**\n")
    lines.append(f"- Scanned rows: **{scanned_rows}**\n")
    lines.append(f"- Hits: **{len(hits_df)}**\n")

    if len(hits_df) == 0:
        lines.append("\n✅ No rows matched Simple Wikipedia heuristics. This strongly suggests that:\n")
        lines.append("- either retrieval/logging only contains ConceptNet-style triples, or\n")
        lines.append("- Simple Wikipedia chunks are not being retrieved, or\n")
        lines.append("- Simple Wikipedia is retrieved but not logged into predictions.csv.\n")
    else:
        lines.append("\n## Hits by (dataset, rag, model)\n")
        grp = hits_df.groupby(["dataset", "rag", "model_raw"]).size().reset_index(name="n_hits")
        lines.append(grp.to_markdown(index=False))
        lines.append("\n## Example hits (first 10)\n")

        preview = hits_df.head(10).copy()
        # shorten long fields
        preview["sentence"] = preview["sentence"].astype(str).str.slice(0, 120)
        preview["context_snippet"] = preview["context_snippet"].astype(str).str.slice(0, 220).str.replace("\n", " ")
        lines.append(preview[[
            "dataset","rag","model_raw","folder","item_id","comb","scope_type","sentence","evidence","context_snippet","predictions_path"
        ]].to_markdown(index=False))

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote: {out_csv}")
    print(f"[OK] wrote: {out_md}")


if __name__ == "__main__":
    main()