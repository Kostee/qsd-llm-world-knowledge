# Data

This directory separates public final benchmarks, deterministic generated files, and local-only historical resources.

## Directory layout

```text
data/
├── public/       # tracked benchmark data
├── generated/    # deterministic derivatives; ignored by Git
└── private/      # local historical/private resources; ignored by Git
```

## Final public benchmarks

### `public/balanced_qsd_440.csv`

Canonical balanced LLM evaluation file.

Properties:

- 440 rows;
- 110 rows per quantifier-combination class;
- 220 surface-scope targets;
- 220 inverse-scope targets;
- gold answer positions: 222 A and 218 B.

Columns:

- `item_id`
- `sentence`
- `Option A`
- `Option B`
- `gold_ans`
- `gold_scope_label`
- `comb`

The preserved historical no-RAG and RAG result folders used these 440 rows directly. They contain no second A/B-reversed row for each balanced item.

The four `comb` values are:

| comb | Pattern | Preferred scope |
|---|---|---|
| 1 | U-E | surface |
| 2 | E-U | surface |
| 3 | E-U | inverse |
| 4 | U-E | inverse |

### `public/pseudosentences_emnlp2026.csv`

Canonical final pseudosentence source benchmark.

Properties:

- 160 source rows;
- 40 source rows per combination class;
- 80 surface-scope targets;
- 80 inverse-scope targets.

Columns:

- `source_idx`
- `sentence`
- `Option A`
- `Option B`
- `gold_ans`
- `gold_scope_label`
- `comb`

The canonical source has the preferred interpretation in Option A. For model evaluation, every source item is expanded into:

1. an `original` A/B ordering;
2. a `flipped` A/B ordering.

## Generating final pseudosentence evaluation files

Run:

```bash
python scripts/prepare_pseudo_paper_final.py
```

The script creates:

```text
data/generated/pseudo_paper_final_model_ready.csv
data/generated/pseudo_paper_final_prompt_only.csv
```

### `pseudo_paper_final_model_ready.csv`

Contains 320 evaluation rows:

- 160 `original`;
- 160 `flipped`;
- 160 rows with `gold_ans=A`;
- 160 rows with `gold_ans=B`;
- 80 rows per combination class;
- 160 surface-scope rows;
- 160 inverse-scope rows.

Columns:

- `idx`
- `source_idx`
- `sentence`
- `Option A`
- `Option B`
- `gold_ans`
- `gold_scope_label`
- `comb`
- `order_variant`

### `pseudo_paper_final_prompt_only.csv`

Contains the same 320 prompt triples without labels or metadata:

- `sentence`
- `Option A`
- `Option B`

## Exact historical verification

During artifact preparation, the generated model-ready and prompt-only pseudosentence files were checked against the historical local files used for the final pseudosentence experiment. Both generated files matched exactly.

The balanced public file was separately checked against the historical 440-item `dataset_for_llms.csv` evaluation content after selecting the evaluation columns used by the LLM runner.

## Important protocol distinction

The historical final experiment did **not** use the same row-expansion behavior for both conditions:

- balanced: 440 evaluation rows;
- pseudosentences: 160 source items expanded to 320 evaluation rows.

This distinction is documented explicitly because the manuscript's prose describes A/B duplication more generally. The artifact prioritizes exact reproduction of the preserved historical execution.

See [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md).

## Generated data

`data/generated/` is ignored by Git because its contents are deterministic derivatives of tracked public data.

Do not edit generated files manually.

## Private and historical data

`data/private/` is ignored by Git and may contain:

- historical paired balanced datasets;
- earlier pseudosentence versions;
- constructed/invented datasets;
- PLM cross-validation resources;
- RAG corpora and FAISS indexes;
- development-time intermediate files.

These local files are not required for the final public balanced or pseudosentence evaluation inputs.

## Public preview files

Historical `*.preview.csv` files remain under `data/public/` for provenance. They are not authoritative inputs for the final paper experiments.

Use:

```text
data/public/balanced_qsd_440.csv
data/public/pseudosentences_emnlp2026.csv
```
