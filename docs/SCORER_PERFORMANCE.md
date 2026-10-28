# Scorer allocation study (28 October 2026)

This measures the standard-library prediction-file **scorer**, not a classifier,
training run, accuracy improvement or general production throughput. The frozen
corpus splits and published model-quality reports are unchanged.

The old path built a list of all CSV row dictionaries, then copied/listed the
rows again inside `score_rows`. Only a label-pair counter is needed for the
metrics. The CLI now validates and consumes the CSV iterator once; `score_rows`
retains counts and the observed label set. Explicit/inferred label order,
zero-support classes in macro F1, legitimate FPR, confusion matrices and exact
input/label byte fingerprints remain unchanged. No report is written before
all records and the label universe pass validation.

## Measured workload

150,000 synthetic rows with a sample ID, two observed labels, and a third
zero-support label. Alternate actual labels and deliberately flip every fifth
prediction. These are generated counting inputs, **not model predictions**.
Use the same exact input/labels for each version. Three baseline/improved pairs
run in fresh Python 3.12.13 processes, with tracemalloc enabled throughout the
CLI call. [All raw measurements and hashes](../evaluation/reports/scorer_allocations_20261028.json).

| Median of three runs | Baseline | Improved |
| --- | ---: | ---: |
| Peak Python traced allocations | 72,537,675 bytes | 20,972,310 bytes |
| Wall time with tracing | 1.046 s | 0.827 s |

Every generated report is byte-identical across all six runs, including its
prediction/label hashes. Tracemalloc excludes native/whole-process RSS; timing
includes tracing overhead, local filesystem effects and one workload on one
machine. This is not a general speedup or memory-cap guarantee. The CSV raw
bytes, decoded text and StringIO snapshot still use memory proportional to the
file size. Inferred arbitrary label universes also grow the counters and final
confusion matrix; only row-dictionary retention was removed.

## Reproduce

From this repository, use Python 3.12. Only pass trusted Python source to
`--scorer`: the benchmark imports it as code. It never runs or trains a model.

```bash
git show 683a8dd6045646cebc014064fa63974e79dd00c7:evaluation/evaluate_predictions.py > /tmp/nlp-scorer-baseline.py
python evaluation/benchmark_scoring.py --scorer /tmp/nlp-scorer-baseline.py --output /tmp/nlp-baseline-measurement.json
python evaluation/benchmark_scoring.py --output /tmp/nlp-improved-measurement.json
```

Use fresh filenames for every measurement; existing output files are refused.
Repeat each version in a fresh process, compare `input_sha256`, `labels_sha256`
and `report_sha256` before comparing allocation/timing values. `--rows` can
change workload size (1–1,000,000); do not compare versions at different sizes.
The helper creates/removes its own synthetic input and report under a temporary
directory and retains only the requested measurement JSON. No private messages
or model artifacts are read. The historical baseline commit may need an
explicit fetch in a shallow clone.
