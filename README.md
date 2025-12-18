# Disentangling Form and World Knowledge in LLM Interpretation
**Evidence from Quantifier Scope Disambiguation (QSD)**

This repository contains code to reproduce experiments for the paper:
*Disentangling Form and World Knowledge in LLM Interpretation: Evidence from Quantifier Scope Disambiguation* (ACL 2026 submission).

## Authors
Jakub Kosterna, Justyna Grudzińska-Zawadowska, Wojciech Borysewicz, Maciej Miecznikowski, Julia Poteralska, Kacper Rutkowski, Jan Kwapisz

## What’s inside
We compare model behavior on QSD under different knowledge conditions:

- **LLM zero-shot (baseline)**: no external context
- **LLM zero-shot + RAG**: dynamic world-knowledge context (ConceptNet + Simple Wikipedia) on the balanced dataset *(placeholder; not implemented yet)*
- **PLM baselines** (fine-tuned): RoBERTa, ERNIE 2.0 (from prior experiments; reused as reference points)

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

- `data/private/MM_balanced_dataset.csv` - full balanced dataset (440 sentence pairs; 880 instances total).
- `data/private/dataset_for_llms.csv` - final evaluation set (440 instances), used as the main input for PLM and LLM experiments on the balanced data.
- `data/private/folds_indices.csv` - indices used for earlier PLM cross-validation runs.
- `data/private/WB_survey_expB2_ABexpanded.csv` - constructed (“invented”) dataset (paired format), used for LLM evaluation without RAG.

Derived files (local):
- `data/private/constructed_for_llms.csv` – one-per-pair selection from the constructed dataset for LLM runs (generated via `scripts/make_constructed_for_llms.py`).

Preview files (tracked):
- `data/public/*.preview.csv`

For details, see `data/README.md`.

## Running experiments

1) Generate the constructed one-per-pair dataset (once)
```
python scripts/make_constructed_for_llms.py ^
  --input data/private/WB_survey_expB2_ABexpanded.csv ^
  --output data/private/constructed_for_llms.csv ^
  --seed 42 ^
  --strict
```

2) Run the unified LLM zero-shot pipeline (single run)
```
python src/llm_zero_shot.py --dataset balanced --limit 5 --model gpt-4o --repeats 5 --seed 42
python src/llm_zero_shot.py --dataset invented --limit all --model qwen:qwen3-8b --repeats 5 --seed 42
```

Where:
- `--dataset` is `balanced` or `invented`
- `--limit` is `0`, `5`, or `all`
- `--model` supports:
  - OpenAI: `gpt-4o`, `gpt-5`, etc.
  - Qwen: `qwen:<model>` (DashScope OpenAI-compatible endpoint)
  - Gemini: `gemini:<model>`
  - OpenRouter: `openrouter:<model>`
 
3) Run batch scripts (Git Bash / Linux / macOS)
```
bash scripts/run_dry0_no_rag.sh
bash scripts/run_sanity5_no_rag.sh
bash scripts/run_all_full_no_rag.sh
```

Before running LLM experiments on the constructed dataset, generate `constructed_for_llms.csv` using `scripts/make_constructed_for_llms.py` (see `data/README.md`).

## Outputs

For LLM experiments, runs are saved under `results/` (not tracked by git). Each run directory contains:
- `predictions.csv` (per-item predictions and per-repeat outputs)
- `metrics.json` and `metrics.csv` (accuracy + breakdowns)
- `config.json` (the exact run configuration)

For PLM experiments, Runs are saved under `runs/` (also not tracked by git). We log:
* `predictions.csv`
* aggregate accuracies (overall, surface vs inverse, per condition)

## License

See `LICENSE`.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).
