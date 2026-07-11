# Target Classification Reproduction Kit

The target task predicts *who/what a toxic or hateful post is about*
(e.g. `individual`, `groups`, `politics`, `race/ethnicity`,
`religion/creed`), independent of the toxicity-label + rationale-span
detection block.

## What's included

```
data/
  vihos_target_fulltext.jsonl           # Vietnamese, derived from ViHOS
scripts/
  train_target_classifier.py            # training/eval script
requirements.txt
```

Each `.jsonl` row has:

```json
{
  "id": "...",
  "split": "train|val|dev|test",
  "source_label": "...",        // original toxicity/hate label
  "target_label": "...",        // gold target class (the training target)
  "label_id": 0,
  "input_text": "...",          // full-text model input
  "source_text": "...",
  "span_strings": ["..."],      // gold rationale span, kept for reference
  "char_spans": [[0, 0]],
  "input_mode": "full_text",
  "span_source": "gold|pred"
}
```

This is the `full_text` input variant (`B1: text -> target`), which was the
best-performing and recommended formulation in the paper's comparison
against span-conditioned variants (`B2`, `B3`).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires a CUDA GPU for reasonable training time; falls back to CPU
automatically if none is available.

## Reproduce: Vietnamese

```bash
python scripts/train_target_classifier.py \
  --data-path data/vihos_target_fulltext.jsonl \
  --model-name roberta-base \
  --output-dir output/target_vn \
  --epochs 3 --batch-size 16 --seed 43
```

Metrics (`accuracy`, `macro_f1`, `weighted_f1`) for validation and test
splits are written to `<output-dir>/metrics_summary.json`, alongside the
saved model/tokenizer checkpoint.

## Data provenance

Derived from [ViHOS](https://github.com/phusroyal/ViHOS) (Vietnamese Hate
and Offensive Spans), re-formatted with the target taxonomy used in the
manuscript (`groups`, `individual`, `politics`, `race/ethnicity`,
`religion/creed`) attached to each full-text input.

Please cite the original ViHOS dataset paper in addition to this
manuscript if you use this data.

## License

Code in this repository is released under the MIT License (see
`LICENSE`). The included data is a derived/relabeled subset provided for
research reproducibility.
