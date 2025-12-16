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
Place full datasets in `data/raw/` (not tracked by git). A small sample may be provided in `data/sample/` for smoke tests.

## Running experiments (high level)

(Commands will be finalized after integrating the unified LLM pipeline.)

* LLM baseline on balanced set:

`python -m src.llm.zero_shot ...`

* LLM + RAG on balanced set:

`python -m src.llm.rag_zero_shot ...`

* LLM baseline on invented set:

`python -m src.llm.zero_shot --dataset invented ...`

## Outputs

Runs are saved under `runs/` (not tracked by git). We log:

* `predictions.csv`

* aggregate accuracies (overall, surface vs inverse, per condition)

## License

See `LICENSE`.

# Citation

If you use this code, please cite the paper (see CITATION.cff).