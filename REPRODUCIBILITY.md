# Reproducibility notes

This document records implementation details established by auditing the preserved historical result folders used during preparation of the public artifact.

It is intentionally explicit about differences between manuscript wording and historical execution.

## 1. Balanced evaluation size

The preserved full balanced no-RAG and balanced RAG result folders contain:

- 440 rows in `predictions.csv`;
- five prediction columns (`pred_run_1` ... `pred_run_5`);
- 222 rows with `gold_ans=A`;
- 218 rows with `gold_ans=B`;
- no paired A/B-reversed duplicate for every balanced row.

Therefore, the historical balanced LLM evaluation underlying the reported tables used the 440-row `dataset_for_llms.csv`-style evaluation file directly.

The public artifact reproduces that input as:

```text
data/public/balanced_qsd_440.csv
```

## 2. Final pseudosentence evaluation size

The preserved final pseudosentence result folders named with `pseudo_paper_final_model_ready` contain:

- 320 rows in `predictions.csv`;
- five prediction columns;
- 160 A and 160 B gold positions;
- 160 valid original/flipped A/B pairs.

The canonical public source has 160 base items:

```text
data/public/pseudosentences_emnlp2026.csv
```

and is deterministically expanded to 320 rows by:

```text
scripts/prepare_pseudo_paper_final.py
```

The generated output was verified against the historical final model-ready file.

## 3. Manuscript wording versus preserved execution

The manuscript states that each datapoint is duplicated with the order of the two interpretations reversed and describes the task formulation as identical across balanced and pseudosentence conditions.

The preserved historical outputs show that A/B-order duplication was present in the final pseudosentence evaluation but not in the balanced LLM evaluation.

For reproducibility, this repository follows the preserved execution:

```text
balanced no-RAG : 440 evaluation rows
balanced RAG    : 440 evaluation rows
pseudosentences : 160 base items -> 320 evaluation rows
```

If the manuscript is revised in the future, the Methods description should be aligned with this historical execution.

## 4. Five repeats and majority voting

For each evaluation row, the unified runner makes five calls by default.

The raw outputs are stored as:

```text
pred_run_1
pred_run_2
pred_run_3
pred_run_4
pred_run_5
```

The runner then computes a row-level majority vote and stores it as:

```text
prediction
```

The Boolean `correct` column compares this majority-vote prediction with `gold_ans`.

Consequently, the standard `metrics.json` accuracy is a **majority-vote accuracy**, not the arithmetic mean of the five repeat-level accuracies.

## 5. Point estimates and variability

Auditing the preserved final result folders shows that the reported overall table point estimates align with majority-vote accuracies after rounding.

The manuscript also reports SD across five stochastic repetitions. Repeat-level variability can be recomputed directly from `pred_run_1` ... `pred_run_5`.

For transparency, the artifact provides:

```bash
python scripts/summarize_paper_results.py
```

which reports both quantities separately:

- `majority_accuracy`;
- `repeat_mean_accuracy`;
- `repeat_sd_sample`.

This avoids presenting a majority-vote point estimate as though it were itself the mean across repeats.

## 6. Smoke-test folders

Runs with `--limit 0` are not model results.

In this mode the runner performs no API calls and fills every repeat prediction with Option A. Such folders exist only to verify the pipeline and should never be included in scientific aggregation.

`--limit 5` is a small API sanity test.

Only `--limit all` folders should be used for final experiment summaries.

## 7. RAG defaults

The current LLM RAG pipeline uses:

```text
RAG_EMBEDDING_MODEL=sentence-transformers/multi-qa-mpnet-base-dot-v1
RAG_TOP_K=3
RAG_MAX_CONTEXT_CHARS=1200
RAG_MAX_PASSAGE_CHARS=400
```

The embedding model used for retrieval must match the embedding model used when the FAISS index was built.

## 8. Historical result provenance

The local audit identified the final result families as:

```text
balanced__limitall__no_rag__repeats5__seed42__...
balanced__limitall__rag__repeats5__seed42__...
pseudo_paper_final_model_ready__limitall__no_rag__repeats5__seed42__...
```

Earlier `invented`, `pseudo_for_llms_emnlp26`, and `pseudo_for_llms_emnlp26_final_model_ready` folders are development/rebuttal-era results and are not the final pseudosentence condition reported in the revised paper.
