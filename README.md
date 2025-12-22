# Disentangling Form and World Knowledge in LLM Interpretation
**Evidence from Quantifier Scope Disambiguation (QSD)**

This repository contains code to reproduce experiments for the paper:
*Disentangling Form and World Knowledge in LLM Interpretation: Evidence from Quantifier Scope Disambiguation* (ACL 2026 submission).

## Authors
Jakub Kosterna, Justyna Grudzińska-Zawadowska, Maciej Miecznikowski, Wojciech Borysewicz, Julia Poteralska, Kacper Rutkowski, Jan Kwapisz

## What’s inside
We compare model behavior on QSD under different knowledge conditions:

- **LLM zero-shot (baseline)**: no external context
- **LLM zero-shot + RAG**: dynamic world-knowledge context retrieved from a local FAISS index
  built over **ConceptNet + Simple Wikipedia** passages
- **PLM baselines** (fine-tuned): RoBERTa / ERNIE 2.0 (from prior experiments; reused as reference points)

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
# Create a local .env (see .env.example) with API keys.
```

## API keys

Create a local `.env` file in the repository root (see `.env.example`).
Depending on the models you run, you may need:

- OpenAI: `OPENAI_API_KEY`
- Qwen (DashScope): `DASHSCOPE_API_KEY` (and optional base URL if you override it)
- Gemini: `GEMINI_API_KEY`
- OpenRouter (e.g., Llama): `OPENROUTER_API_KEY`

Note: `.env` is not tracked by git.

## Data

Full datasets are kept locally under `data/private/` (not tracked by git).
Small preview files are provided in `data/public/`.

Expected full files (local):

- `data/private/MM_balanced_dataset.csv` – full balanced dataset (paired format).
- `data/private/dataset_for_llms.csv` – final balanced evaluation set (single instance per pair; contains comb).
- `data/private/WB_survey_expB2_ABexpanded.csv` – constructed ("invented") dataset (paired format; contains comb).
- `data/private/invented_for_llms.csv` – final invented evaluation set (single instance per pair; contains comb).
- `data/private/folds_indices.csv` – indices used for earlier PLM cross-validation runs.

Preview files (tracked):
- `data/public/*.preview.csv`

For details, see `data/README.md`.

## RAG: build the retrieval corpus + FAISS index (once)

RAG runs reuse artifacts produced by `src/build_rag_advanced.py`.
This step builds a passage corpus (ConceptNet + Simple Wikipedia), cleans & chunks it, deduplicates,
embeds passages (SentenceTransformers), and creates a FAISS index.

```bash
python src/build_rag_advanced.py --output-dir data/private/rag_corpus
```

Artifacts written to `--output-dir` include (by default):
- `wiki_passages.txt` - text passages
- `wiki_passages.pkl` - list of passages (List[str])
- `wiki_passages.faiss` - FAISS index over normalized embeddings

You can control runtime retrieval via env vars:
- `RAG_DIR` (default: `data/private/rag_corpus`)
- `RAG_PASSAGES_BASENAME` (default: `wiki_passages`)
- `RAG_EMBEDDING_MODEL` (must match index build)
- `RAG_TOP_K` (default: 3)
- `RAG_MAX_CONTEXT_CHARS`, `RAG_MAX_PASSAGE_CHARS`

## Running experiments

1) (Optional) Build the invented evaluation set (once)

If you need to regenerate `invented_for_llms.csv` from the paired invented dataset:

Generate the constructed one-per-pair dataset (once)
```
python scripts/make_constructed_for_llms.py ^
  --input data/private/WB_survey_expB2_ABexpanded.csv ^
  --output data/private/constructed_for_llms.csv ^
  --seed 42 ^
  --strict
```

2) Run the unified LLM zero-shot pipeline (single run)

No-RAG
```bash
python src/llm_zero_shot.py --dataset balanced --limit 5 --model gpt-4o --repeats 5 --seed 42
python src/llm_zero_shot.py --dataset invented --limit all --model qwen:qwen3-8b --repeats 5 --seed 42
```

With RAG
```bash
python src/llm_zero_shot.py --dataset balanced --limit all --rag --model gpt-4o --repeats 5 --seed 42
```

Where:
- `--dataset` is `balanced` or `invented`
- `--limit` is `0`, `5`, or `all`
- `--model` supports:
  - OpenAI: `gpt-4o`, `gpt-4-0613`, `o3-mini`...
  - Qwen: `qwen:<model>` (DashScope OpenAI-compatible endpoint)
  - Gemini: `gemini:<model>`
  - OpenRouter: `openrouter:<model>`
 
3) Run batch scripts (Git Bash / Linux / macOS)
```
bash scripts/run_dry0_no_rag.sh
bash scripts/run_sanity5_no_rag.sh
bash scripts/run_all_full_no_rag.sh
bash scripts/run_all_full_rag.sh
```

## Outputs

LLM runs are saved under `results/` (not tracked by git). Each run directory contains:
- `predictions.csv` - per-item predictions (per repeat) + fields needed for aggregation
- `metrics.json` and `metrics.csv` (accuracy + breakdowns)
- `config.json` - the exact run configuration

PLM experiments are saved under `runs/` (also not tracked by git). We log:
* `predictions.csv`
* aggregate accuracies (overall, surface vs inverse, per condition)

## Post-processing & utilities (scripts/)

- `scripts/summarize_results.py`
Aggregates all run folders under results/ into results/metrics_summary.csv.

- `scripts/backfill_invented_comb_and_metrics.py` (one-off migration)
Backfills missing comb into:

	- `data/private/invented_for_llms.csv` (if needed),

	- `results/invented*/predictions.csv`, and recomputes `metrics.json` / `metrics.csv` so invented runs have per-combination metrics
consistent with balanced runs.

- `scripts/run_dry0_no_rag.sh`
Quick smoke test (no real API calls; --limit 0).

- `scripts/run_sanity5_no_rag.sh`
Small sanity run (`--limit 5`) for a subset of models/datasets.

- `scripts/run_all_full_no_rag.sh`
Full no-RAG sweep across configured models and datasets.

- `scripts/run_all_full_rag.sh`
Full RAG sweep across configured models (typically used for balanced).

- `scripts/make_constructed_for_llms.py`
Creates the invented evaluation set (single instance per pair) from the paired invented dataset.

- `scripts/select_random_from_pairs.py`
Helper for selecting one item from paired data (used in dataset preparation utilities).

- `scripts/generate_indices_for_cross.py`
Generates indices for cross-validation / split utilities (legacy support for earlier PLM experiments).

## License

See `LICENSE`.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).
