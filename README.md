# Target-aware toxic content detection — reproduction package

Code and data to reproduce the experimental tables of the PLOS ONE manuscript
*Evaluating Evidence Grounding for Target-Aware Toxic Content Detection* (PONE-D-26-35222).

Two blocks are covered:

* **MRE** — a multitask rationale extractor (XLM-RoBERTa + sentence-toxicity head + BIO span head
  + rationale-guided fusion + consistency regularisation) that produces the predicted spans used in
  the third input condition.
* **Target identification** — five-way classification (`individual`, `groups`, `politics`,
  `race/ethnicity`, `religion/creed`) under three input formulations:
  **B1** sentence only, **B2** gold spans marked, **B3** predicted spans marked.

## Contents

```
data/
  vihos_prepared.jsonl                   # toxicity + span training data for the MRE (11,056 rows)
  vihos_target_fulltext.jsonl            # B1: 5,360 toxic comments, no markers
  vihos_target_highlighted.jsonl         # B2: same rows, gold spans marked with [HATE_SPAN]
  vihos_target_highlighted_pred.jsonl    # B3: same rows, model-predicted spans marked
scripts/
  train_mre.py                           # MRE training entry point
  train_multitask_unified.py            # trainer + model (dataset, losses, loop)
  model_multitask.py                     # XLMRMultiTask model definition
  eval_mre.py                            # sentence + span evaluation (use --only_stage1)
  train_target_classifier.py             # target-identification training/eval
  eval_target_perclass.py                # per-class metrics + confusion matrix
  config_loader.py                       # package-local paths used by eval_mre.py
configs/
  mre.yaml                               # Table 6 hyperparameters + seeds
  target.yaml                            # Table 7 hyperparameters, conditions, seeds
  seeds.tsv                              # flat seed table (one row per cell)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU is expected for the reported timings (the MRE takes ~30 min per run on an RTX 4060 Ti;
target identification takes 5–9 min per run). Both scripts fall back to CPU, which is far slower.

## Block 1 — MRE (Table 6)

Train one run:

```bash
python scripts/train_mre.py \
  --data_path data/vihos_prepared.jsonl \
  --output_dir output/mre/seed42 \
  --model_name xlm-roberta-base \
  --epochs 10 --batch_size 16 --learning_rate 2e-5 --max_len 256 \
  --alpha 5.0 --use_fusion 1 --use_consistency 1 --lambda_const 0.1 \
  --use_class_weights 1 --best_metric val_f1 --seed 42
```

Evaluate sentence-level and span-level metrics on the test split:

```bash
python scripts/eval_mre.py --provider local \
  --model_path output/mre/seed42/best_model.pth \
  --only_stage1 --output mre_seed42.json
```

`--only_stage1` reports the encoder's own predictions; the LLM-judge stages of the original
pipeline are not part of the paper's reported numbers and are not needed here.

Reference values (mean ± sd over seeds 42–46, test split): classification accuracy
0.9047 ± 0.0074, classification macro-F1 0.9045, span accuracy 0.9120 ± 0.0033, span macro-F1
0.8596 ± 0.0043.

**Ablation used in the revision.** The loss is `L = L_cls + α·L_BIO + λ·L_cons`, and the ablation
switches two flags: `--alpha 0 --use_consistency 0` (sentence head only), `--alpha 5.0
--use_consistency 0` (adds the BIO loss), `--alpha 5.0 --use_consistency 1 --lambda_const 0.1`
(full model). Keep `--use_fusion 1` in all three to change the loss only, not the architecture.

## Block 2 — target identification (Table 7)

Each condition is a separate data file, everything else is identical:

```bash
for cond in B1:vihos_target_fulltext B2:vihos_target_highlighted B3:vihos_target_highlighted_pred; do
  tag=${cond%%:*}; file=${cond#*:}
  python scripts/train_target_classifier.py \
    --data-path data/$file.jsonl \
    --model-name xlm-roberta-base \
    --output-dir output/target/xlmr_${tag}_seed201 \
    --seed 201 --epochs 10 --early-stop-patience 3 --early-stop-min-delta 0.0
done
```

Backbones and seeds used in the paper (`configs/seeds.tsv`): XLM-RoBERTa-Base 201–206,
PhoBERT-Base-v2 42–47, BARTpho-word 42–47, ViSoBERT 42–46. For ViSoBERT use
`--model-name uitnlp/visobert`; for BARTpho `--model-name vinai/bartpho-word`.

Reference values (macro-F1, mean ± sd, test split):

| Backbone | B1 | B2 | B3 |
|---|---|---|---|
| XLM-RoBERTa-Base | 0.6581 ± 0.0171 | 0.4434 ± 0.2383 | 0.4553 ± 0.2523 |
| PhoBERT-Base-v2 | 0.6966 ± 0.0383 | 0.6384 ± 0.0829 | 0.6393 ± 0.0193 |
| BARTpho-word | 0.6598 ± 0.0160 | 0.6392 ± 0.0213 | 0.6074 ± 0.0308 |
| ViSoBERT | 0.7216 ± 0.0128 | 0.7067 ± 0.0161 | 0.6206 ± 0.0159 |

Note on B3: the marked spans come from the MRE for the **test split only** (531 rows); training and
development rows keep their gold spans, because the MRE is trained on those same rows. The field
`span_source` in the jsonl records this per row (`gold` or `pred`).

## Per-class results and confusion matrices

```bash
python scripts/eval_target_perclass.py \
  --ckpt output/target/xlmr_B1_seed201 \
  --data data/vihos_target_fulltext.jsonl \
  --out perclass/xlmr_B1_seed201.json
```

The output JSON contains accuracy, macro/weighted F1, per-class precision/recall/F1/support and the
5×5 confusion matrix. Averaging the per-seed files gives the per-class table reported in the
revision's supporting information.

## Expected runtime

| Block | Runs | Per run | Total |
|---|---|---|---|
| MRE (Table 6) | 5 | ~30 min | ~2.5 h |
| Target identification (Table 7) | 54 (4 backbones × 3 conditions × seeds) | 5–9 min | ~6 h |
| Per-class evaluation | 54 | seconds | minutes |

## Data provenance and licence

The comments and their toxic spans come from **ViHOS** (Hoang et al., EACL 2023), released under the
MIT licence; please cite it alongside this package. The five target categories are defined by the
annotation guideline reproduced in Appendix A of the manuscript, and the protocol used to attach
them is described in Section 4.1 of the manuscript. The three target jsonl files are the exact files
consumed by the training scripts; `data/vihos_target_fulltext.jsonl` is byte-identical to the file
used in the published experiments.

Code is released under the MIT licence (see `LICENSE`). The derived data remain subject to the terms
of the original ViHOS release.

## Citation

If you use this package, please cite the manuscript (PONE-D-26-35222) and ViHOS.
