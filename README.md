# Disentangling Form and World Knowledge in LLM Interpretation

**Evidence from Quantifier Scope Disambiguation (QSD)**

This repository contains the code and public evaluation data associated with:

> **Disentangling Form and World Knowledge in LLM Interpretation: Evidence from Quantifier Scope Disambiguation**

The current repository state corresponds to the revised 2026 submission and its reproducibility artifact.

## Authors

Jakub Kosterna, Justyna Grudzińska-Zawadowska, Maciej Miecznikowski, Wojciech Borysewicz, Julia Poteralska, Kacper Rutkowski, Jan Kwapisz

## Overview

We use Quantifier Scope Disambiguation (QSD) as a controlled probe of how large language models combine formal interpretive cues with lexical-semantic and world knowledge.

The final LLM experiments evaluate seven models:

- GPT-5.1
- GPT-4o
- GPT-4o-mini
- Qwen-Max
- Qwen3-8B
- Llama 3.1 70B
- Llama 3.1 8B

The repository contains three main reproducibility entry points:

1. balanced QSD, no RAG;
2. final pseudosentences, no RAG;
3. balanced QSD with RAG.

## Main datasets

### Balanced QSD benchmark

`data/public/balanced_qsd_440.csv`

- 440 evaluation items;
- 110 items per quantifier-combination class;
- 220 surface-scope targets;
- 220 inverse-scope targets;
- historical LLM runs used these 440 rows directly.

The preserved historical balanced result folders contain 440 evaluation rows, with 222 gold A labels and 218 gold B labels. They do **not** contain a second A/B-reversed row for every item.

### Final pseudosentence benchmark

`data/public/pseudosentences_emnlp2026.csv`

- 160 canonical source items;
- 40 items per quantifier-combination class;
- 80 surface-scope targets;
- 80 inverse-scope targets.

For the final pseudosentence model evaluation, each source item is represented in two answer-order variants: original and A/B-flipped. The deterministic preprocessing step therefore creates 320 evaluation rows.

Generate the model-ready file with:

```bash
python scripts/prepare_pseudo_paper_final.py
```

This writes:

```text
data/generated/pseudo_paper_final_model_ready.csv
data/generated/pseudo_paper_final_prompt_only.csv
```

The generated 320-row files were verified against the historical files used for the final pseudosentence runs.

For dataset details, see [`data/README.md`](data/README.md).

## Important reproducibility note

The preserved historical outputs reveal two implementation details that are important for exact reproduction:

- balanced no-RAG and balanced RAG runs used the 440-row canonical balanced evaluation file;
- final pseudosentence runs used 160 base items expanded to 320 A/B-order evaluation rows.

The manuscript describes A/B-order duplication more generally across conditions. The repository follows the **historical execution that produced the reported results**. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the audit trail and scoring details.

## Scoring protocol

For every evaluation row, `src/llm_zero_shot.py` makes five model calls by default (`--repeats 5`).

The result file stores:

```text
pred_run_1
pred_run_2
pred_run_3
pred_run_4
pred_run_5
prediction
correct
```

`prediction` is the majority vote across the five calls. The standard `metrics.json` point estimate is computed from this majority-vote prediction.

To inspect both the majority-vote point estimate and variability across the five repeat-level accuracies, run:

```bash
python scripts/summarize_paper_results.py
```

The script writes:

```text
results/paper_metrics_summary.csv
```

and reports, separately:

- majority-vote accuracy;
- mean accuracy across the five repeat columns;
- sample SD across the five repeat columns;
- surface/inverse breakdowns;
- combination-specific breakdowns.

This separation is intentional: it makes the historical scoring convention explicit rather than conflating the majority-vote point estimate with the mean across repeats.

## Quantifier-combination classes

| Type | Pattern | Preferred reading |
|---|---|---|
| I | U-E | surface |
| II | E-U | surface |
| III | E-U | inverse |
| IV | U-E | inverse |

## Setup

Python 3.12 is the tested artifact environment.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install the artifact dependencies:

```bash
pip install -r requirements.txt
```

Create a local `.env` file from `.env.example`.

## API keys

The final seven-model paper sweep uses:

```text
OPENAI_API_KEY
QWEN_API_KEY
OPENROUTER_API_KEY
```

The unified runner also supports Gemini through:

```text
GOOGLE_API_KEY
```

Gemini is not required for the final seven-model sweep and its LangChain integration is therefore not part of the pinned core artifact environment.

The `.env` file is ignored by Git.

## Reproducing the final LLM experiments

### Balanced QSD — no RAG

Single-model example:

```bash
python src/llm_zero_shot.py \
  --dataset balanced \
  --limit all \
  --model gpt-4o \
  --repeats 5 \
  --seed 42
```

Final seven-model sweep:

```bash
bash scripts/run_all_full_no_rag_balanced_paper_final.sh
```

The `balanced` selector resolves to:

```text
data/public/balanced_qsd_440.csv
```

### Final pseudosentences — no RAG

Prepare the final evaluation data:

```bash
python scripts/prepare_pseudo_paper_final.py
```

or use the batch runner, which performs preparation automatically:

```bash
bash scripts/run_all_full_no_rag_pseudo_paper_final.sh
```

### Balanced QSD — RAG

Build the ConceptNet + Simple Wikipedia retrieval corpus and FAISS index:

```bash
python src/build_rag_advanced.py --output-dir data/private/rag_corpus
```

The current LLM retrieval defaults are:

```text
RAG_EMBEDDING_MODEL=sentence-transformers/multi-qa-mpnet-base-dot-v1
RAG_TOP_K=3
RAG_MAX_CONTEXT_CHARS=1200
RAG_MAX_PASSAGE_CHARS=400
```

Single-model example:

```bash
python src/llm_zero_shot.py \
  --dataset balanced \
  --limit all \
  --rag \
  --model gpt-4o \
  --repeats 5 \
  --seed 42
```

Final seven-model sweep:

```bash
bash scripts/run_all_full_rag_paper_final.sh
```

## Smoke tests

`--limit 0` is a plumbing test only. It makes no API calls and fills predictions with Option A, so its accuracy is **not** a model result.

Example:

```bash
python src/llm_zero_shot.py --dataset balanced --limit 0 --model gpt-4o --repeats 5 --seed 42
```

Use `--limit 5` for a small real-API sanity run and `--limit all` for the full experiment.

## Outputs

LLM runs are written under:

```text
results/
```

This directory is ignored by Git.

A run directory contains:

```text
predictions.csv
metrics.json
metrics.csv
config.json
```

For paper-oriented aggregation, prefer:

```bash
python scripts/summarize_paper_results.py
```

The older `scripts/summarize_results.py` is retained for historical compatibility.

## Repository structure

```text
.
├── data/
│   ├── public/
│   │   ├── balanced_qsd_440.csv
│   │   ├── pseudosentences_emnlp2026.csv
│   │   └── *.preview.csv
│   ├── generated/              # deterministic, ignored
│   └── private/                # local-only historical/RAG resources, ignored
├── scripts/
│   ├── prepare_pseudo_paper_final.py
│   ├── summarize_paper_results.py
│   ├── run_all_full_no_rag_balanced_paper_final.sh
│   ├── run_all_full_no_rag_pseudo_paper_final.sh
│   ├── run_all_full_rag_paper_final.sh
│   └── ...
├── src/
│   ├── llm_zero_shot.py
│   ├── build_rag_advanced.py
│   └── plm/
├── REPRODUCIBILITY.md
├── README.md
├── requirements.txt
└── CITATION.cff
```

## Legacy and auxiliary code

The repository retains utilities from earlier project stages, including earlier constructed/invented datasets, rebuttal/statistical scripts, older batch scripts, and fine-tuned PLM baselines.

They are kept for provenance but are not the primary entry points for the final seven-model LLM experiments.

## License and source data

Repository code is covered by [`LICENSE`](LICENSE).

The balanced benchmark is derived in part from the scope-ambiguity resources introduced by Kamath et al. (2024). Please preserve source attribution when redistributing or adapting the data.

## Citation

If you use this repository, please cite the associated paper. Citation metadata is provided in [`CITATION.cff`](CITATION.cff).
