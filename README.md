# Disentangling Form and World Knowledge in LLM Interpretation
**Evidence from Quantifier Scope Disambiguation (QSD)**

This repository contains code to reproduce experiments for the paper:
*Disentangling Form and World Knowledge in LLM Interpretation: Evidence from Quantifier Scope Disambiguation* (ACL 2026 submission).

## Authors
Jakub Kosterna, Justyna Grudzińska-Zawadowska, Wojciech Borysewicz, Maciej Miecznikowski, Julia Poteralska, Kacper Rutkowski, Jan Kwapisz

## What’s inside
We compare model behavior on QSD under different knowledge conditions:

- **LLM zero-shot (baseline)**: no external context
- **LLM zero-shot + RAG**: dynamic world-knowledge context (ConceptNet + Simple Wikipedia) on the balanced dataset
- **PLM baselines** (fine-tuned): RoBERTa, ERNIE 2.0 (from prior experiments; reused as reference points)

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
# Create a local .env (see .env.example) with API keys.
```

## Data

Full datasets are kept locally under `data/private/` (not tracked by git).
Small preview files are provided in `data/public/`.

Expected full files (local):

- `data/private/MM_balanced_dataset.csv` - full balanced dataset (440 sentence pairs; 880 instances total).
- `data/private/dataset_for_llms.csv` - final evaluation set (440 instances), used as the main input for PLM and LLM experiments on the balanced data.
- `data/private/folds_indices.csv` - indices used for earlier PLM cross-validation runs.
- `data/private/WB_survey_expB2_ABexpanded.csv` - constructed (“invented”) dataset (paired format), used for LLM evaluation without RAG.

Derived files (local):
- `data/private/constructed_for_llms.csv` – one-per-pair selection from the constructed dataset for LLM runs (generated via `scripts/make_constructed_for_llms.py`).

Preview files (tracked):
- `data/public/*.preview.csv`

For details, see `data/README.md`.

## Running experiments (high level)

(Commands will be finalized after integrating the unified LLM pipeline.)

* LLM baseline on balanced set:
`python -m src.llm.zero_shot ...`

* LLM + RAG on balanced set:
`python -m src.llm.rag_zero_shot ...`

* LLM baseline on invented set:
`python -m src.llm.zero_shot --dataset invented ...`

Before running LLM experiments on the constructed dataset, generate `constructed_for_llms.csv` using `scripts/make_constructed_for_llms.py` (see `data/README.md`).

## Outputs

Runs are saved under `runs/` (not tracked by git). We log:
* `predictions.csv`
* aggregate accuracies (overall, surface vs inverse, per condition)

## License

See `LICENSE`.

## Citation

If you use this code, please cite the paper (see CITATION.cff).
