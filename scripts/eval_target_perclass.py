#!/usr/bin/env python3
"""Per-class evaluation of a saved target-identification checkpoint.

Why this exists: reviewers of PONE-D-26-35222 asked for per-class results and
confusion matrices, especially for religion/creed (73 rows in the full set, 6 in
test). The training script only recorded accuracy / macro-F1, so this script
re-loads an existing checkpoint and recomputes the finer-grained report without
retraining anything.

Usage:
    python scripts/perclass_target_eval.py \
        --ckpt experiments/vietnamese/models/target_xlmr_B1_v2_seed201 \
        --data experiments/vietnamese/data/vihos_target_fulltext_v1.jsonl \
        --out  experiments/table_seed_runs/perclass/xlmr_B1_seed201.json
"""

import argparse
import inspect
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["groups", "individual", "politics", "race/ethnicity", "religion/creed"]


def load_rows(path, split):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("split") != split:
                continue
            rows.append(rec)
    return rows


@torch.no_grad()
def predict(model, tok, texts, max_len, batch_size, device):
    preds = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        enc = tok(
            chunk,
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        ).to(device)
        # BARTpho checkpoints load as MBartForSequenceClassification, whose forward()
        # rejects token_type_ids; keep only the tensors the model actually accepts.
        accepted = set(inspect.signature(model.forward).parameters)
        enc = {k: v for k, v in enc.items() if k in accepted}
        logits = model(**enc).logits
        preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt).to(device).eval()

    rows = load_rows(args.data, args.split)
    texts = [r["input_text"] for r in rows]
    gold = [int(r["label_id"]) for r in rows]

    preds = predict(model, tok, texts, args.max_len, args.batch_size, device)

    gold_a = np.asarray(gold)
    pred_a = np.asarray(preds)
    cm = confusion_matrix(gold_a, pred_a, labels=list(range(len(LABELS))))

    per_class = {}
    for idx, name in enumerate(LABELS):
        tp = int(cm[idx, idx])
        fp = int(cm[:, idx].sum() - tp)
        fn = int(cm[idx, :].sum() - tp)
        support = int(cm[idx, :].sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
            "correct": tp,
        }

    payload = {
        "ckpt": str(args.ckpt),
        "data": str(args.data),
        "split": args.split,
        "n": int(len(rows)),
        "accuracy": round(float(accuracy_score(gold_a, pred_a)), 4),
        "macro_f1": round(float(f1_score(gold_a, pred_a, average="macro", labels=list(range(len(LABELS))), zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(gold_a, pred_a, average="weighted", labels=list(range(len(LABELS))), zero_division=0)), 4),
        "labels": LABELS,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "gold_distribution": {LABELS[i]: int((gold_a == i).sum()) for i in range(len(LABELS))},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("ckpt", "n", "accuracy", "macro_f1")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
