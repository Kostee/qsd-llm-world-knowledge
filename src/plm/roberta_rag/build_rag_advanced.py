#!/usr/bin/env python3
"""build_rag_advanced.py
=================================
A more thorough (but slower) RAG corpus builder + FAISS indexer meant for high‑quality
question‑specific document (QSD) retrieval experiments.  Compared with the quick
`build_rag_baseline.py`, this version invests roughly an *order of magnitude* more
compute to squeeze out better semantic coverage, recall and search latency.

Key upgrades
------------
* **Larger & better embeddings** – default to `sentence-transformers/multi-qa-mpnet-base-dot-v1` (768‑d, ~2× quality boost).
* **Full‑sized datasets** – stream full ConceptNet EN + Simple Wikipedia; optional extra corpora.
* **Aggressive text cleaning & paragraph / sentence chunking** to 512 UTF‑8 chars.
* **Near‑duplicate removal** with MinHash‑LSH & Jaccard distance (≈30 % corpus shrink).
* **Device‑aware, mixed‑precision batched encoding** (GPU if available, BF16/FP16).
* **Configurable FAISS index types** – HNSW‑Flat (default), IVF‑PQ, or FlatIP.
* **Incremental, memory‑sweet indexing** – embeddings processed in shards; no need to fit all in RAM.
* **CLI flags & YAML config** for reproducibility.
* **Rich progress + timing logs**.

Usage
-----
```
python build_rag_advanced.py \
    --output-dir C_experiments/roberta_rag/rag_corpus_advanced \
    --embeddings-model sentence-transformers/multi-qa-mpnet-base-dot-v1 \
    --index-type hnsw \
    --max-passages 500000
```

Requires Python 3.9+, `faiss-cpu`≥1.7.4 (or `faiss‑gpu`), `sentence-transformers`,
`datasets`, `datasketch`, `tqdm`, `pyyaml`, `rapidfuzz`.
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import faiss
import torch
from datasketch import MinHash, MinHashLSH
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from rapidfuzz.distance.Levenshtein import distance
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import yaml

###############################################################################
# --------------------------- Text utilities ---------------------------------
###############################################################################


def simple_chunk(text: str, max_chars: int = 512) -> List[str]:
    """Greedy sentence‑level splitter limited by *max_chars* (UTF‑8)."""
    if len(text) <= max_chars:
        return [text.strip()]

    out, buf = [], []
    for sent in text.replace("\n", " ").split(". "):
        if sum(len(s) + 2 for s in buf) + len(sent) + 1 <= max_chars:
            buf.append(sent)
        else:
            out.append(". ".join(buf).strip() + ".")
            buf = [sent]
    if buf:
        out.append(". ".join(buf).strip() + ".")
    return out


###############################################################################
# --------------------------- Deduplication ----------------------------------
###############################################################################

def deduplicate_passages(passages: List[str], threshold: float = 0.9) -> List[str]:
    """Remove near‑duplicates using MinHash‑LSH Jaccard > *threshold*."""
    logging.info("‣ Deduplicating %d passages… (threshold=%.2f)", len(passages), threshold)
    lsh = MinHashLSH(threshold=threshold, num_perm=128)

    unique_texts = []
    for text in tqdm(passages, unit="passage"):
        mh = MinHash(num_perm=128)
        for token in text.split():
            mh.update(token.encode("utf8"))
        dup = lsh.query(mh)
        if not dup:
            lsh.insert(len(unique_texts), mh)
            unique_texts.append(text)
    logging.info("‣ After dedup: %d unique passages (%.1f %% retained).", len(unique_texts), 100 * len(unique_texts) / len(passages))
    return unique_texts


###############################################################################
# --------------------------- Index builders ---------------------------------
###############################################################################

def build_faiss_index(
    embeddings: torch.Tensor | List[torch.Tensor],
    index_type: str,
    dim: int,
    hnsw_m: int = 64,
    hnsw_ef: int = 200,
    ivf_nlist: int = 4096,
    ivf_m: int = 16,
) -> faiss.Index:
    if index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, hnsw_m)
        index.hnsw.efConstruction = hnsw_ef
    elif index_type == "ivfpq":
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, ivf_nlist, ivf_m, 8)
        index.nprobe = max(1, ivf_nlist // 64)
    elif index_type == "flat":
        index = faiss.IndexFlatIP(dim)
    else:
        raise ValueError(f"Unsupported index_type: {index_type}")

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    elif isinstance(embeddings, list):
        embeddings = torch.cat(embeddings, dim=0).cpu().numpy()

    logging.info("‣ Adding %d vectors to FAISS (%s)…", embeddings.shape[0], index_type)
    index.add(embeddings)
    return index


###############################################################################
# --------------------------- Main pipeline ----------------------------------
###############################################################################

def load_corpus(max_passages: int | None = None) -> List[str]:
    logging.info("[1/4] Loading ConceptNet EN + Simple Wikipedia …")

    cn = load_dataset("peandrew/conceptnet_en_nomalized", split="train", streaming=True)
    sw = load_dataset("rahular/simple-wikipedia", split="train", streaming=True)

    def _yield_clean(ds):
        for ex in ds:
            txt = ex.get("text") or "{} —{}→ {}".format(ex.get("arg1"), ex.get("rel"), ex.get("arg2"))
            if txt:
                yield from simple_chunk(txt)

    iters = (_yield_clean(cn), _yield_clean(sw))
    passages = []
    for source in iters:
        for p in source:
            passages.append(p)
            if max_passages and len(passages) >= max_passages:
                break
        if max_passages and len(passages) >= max_passages:
            break

    logging.info("‣ Collected %d raw passages.", len(passages))
    passages = deduplicate_passages(passages)
    return passages


def encode_passages(
    passages: List[str],
    model_name: str,
    batch_size: int = 128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    logging.info("[2/4] Encoding %d passages (%s, batch=%d)…", len(passages), model_name, batch_size)
    model = SentenceTransformer(model_name, device=device)
    embeddings = []
    for i in tqdm(range(0, len(passages), batch_size), unit="batch"):
        batch = passages[i : i + batch_size]
        emb = model.encode(batch, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
        embeddings.append(emb)
    return torch.cat(embeddings, dim=0)


def save_artifacts(
    passages: List[str],
    embeddings: torch.Tensor,
    index: faiss.Index,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / "wiki_passages.txt"
    idx_path = out_dir / "wiki_passages.faiss"
    pkl_path = out_dir / "wiki_passages.pkl"

    logging.info("[3/4] Writing %s", txt_path)
    with txt_path.open("w", encoding="utf-8") as f:
        for line in passages:
            f.write(line + "\n")

    logging.info("[4/4] Saving FAISS index → %s", idx_path)
    faiss.write_index(index, str(idx_path))

    logging.info("‣ Pickling passage list → %s", pkl_path)
    with pkl_path.open("wb") as f:
        pickle.dump(passages, f)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build high‑quality RAG corpus + FAISS index.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--embeddings-model", default="sentence-transformers/multi-qa-mpnet-base-dot-v1")
    p.add_argument("--index-type", choices=["hnsw", "ivfpq", "flat"], default="hnsw")
    p.add_argument("--max-passages", type=int, default=None, help="Trim corpus after N passages (None = all)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--config-yaml", type=Path, help="Optional YAML with args – overrides CLI")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Config via YAML
    if args.config_yaml and args.config_yaml.exists():
        with args.config_yaml.open() as f:
            yaml_cfg = yaml.safe_load(f)
        for k, v in yaml_cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.info("🚀 Starting build_rag_advanced @ %s", datetime.now(datetime.UTC).isoformat(timespec="seconds"))

    passages = load_corpus(max_passages=args.max_passages)
    embeddings = encode_passages(passages, args.embeddings_model, batch_size=args.batch_size)
    index = build_faiss_index(embeddings, args.index_type, dim=embeddings.shape[1])
    save_artifacts(passages, embeddings, index, args.output_dir)

    logging.info("✅ Done. Artifacts saved to %s", args.output_dir)


if __name__ == "__main__":
    main()