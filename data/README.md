# Data

This repository includes the datasets used in our QSD experiments.

## Files

### `MM_balanced_dataset.csv`
**Full balanced dataset** consisting of **440 sentence pairs** (i.e., 880 instances in total).  
This file is the *paired source dataset* used to construct the final evaluation set.

### `dataset_for_llms.csv`
**Final evaluation dataset (440 instances)** used as the main benchmark in the project - i.e., the "single source of truth" for the new experiments (the core of the paper).

This file was created in the earlier workflow using the scripts:
- `scripts/generate_indices_for_cross.py`
- `scripts/select_random_from_pairs.py`

In short, the procedure:
1) generated a fixed set of indices / split configuration for the previous experimental setup, and  
2) selected **exactly one sentence from each pair** in `MM_balanced_dataset.csv`, producing a deterministic set of **440 sentences**.

We reuse `dataset_for_llms.csv` as the primary dataset for the current **LLM zero-shot** experiments (baseline vs. RAG-enhanced), ensuring all models are evaluated on exactly the same inputs.

### `folds_indices.csv`
Pre-generated indices/splits used for the earlier **PLM cross-validation** runs.

## Notes
- The "paired" file (`MM_balanced_dataset.csv`) is kept for transparency and reproducibility of the selection procedure.
- The main benchmark for the current paper is `dataset_for_llms.csv` (440 instances).