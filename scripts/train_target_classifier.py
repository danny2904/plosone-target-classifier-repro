import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Vietnamese target classifier on full-text target-labeled data. "
        "Reproduction script for the PLOS ONE manuscript's Block B (text -> target) model."
    )
    parser.add_argument(
        "--data-path",
        default="data/vihos_target_fulltext.jsonl",
        help="Prepared target-classification JSONL path (see data/).",
    )
    parser.add_argument(
        "--model-name",
        default="roberta-base",
        help="HF encoder checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/target_model",
        help="Output directory for checkpoints and metrics.",
    )
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-class-weights", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


class TargetDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer, max_len):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        enc = self.tokenizer(
            row["input_text"],
            truncation=True,
            max_length=self.max_len,
            padding=False,
        )
        enc["labels"] = int(row["label_id"])
        return enc


def compute_metrics(logits, labels):
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }


def maybe_limit(rows, limit):
    if limit and limit > 0:
        return rows[:limit]
    return rows


def build_class_weights(train_rows, num_labels):
    counts = np.zeros(num_labels, dtype=np.float32)
    for row in train_rows:
        counts[int(row["label_id"])] += 1.0
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_labels * counts)
    return torch.tensor(weights, dtype=torch.float)


def collate_batch(batch, tokenizer):
    labels = torch.tensor([item.pop("labels") for item in batch], dtype=torch.long)
    padded = tokenizer.pad(batch, padding=True, return_tensors="pt")
    padded["labels"] = labels
    return padded


def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_steps = 0
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k not in ("labels", "token_type_ids")}
            outputs = model(**inputs)
            logits = outputs.logits
            loss = nn.CrossEntropyLoss()(logits, labels)
            total_loss += loss.item()
            total_steps += 1
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(logits_np, labels_np)
    metrics["loss"] = total_loss / max(total_steps, 1)
    return metrics


def main():
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    rows = load_jsonl(args.data_path)
    label_names = sorted({row["target_label"] for row in rows})
    num_labels = len({int(row["label_id"]) for row in rows})

    split_names = {row["split"] for row in rows}
    val_split_name = "val" if "val" in split_names else ("dev" if "dev" in split_names else None)
    if val_split_name is None:
        raise ValueError(f"No validation split found in {args.data_path}; expected 'val' or 'dev'.")

    train_rows = maybe_limit([row for row in rows if row["split"] == "train"], args.max_train_samples)
    val_rows = maybe_limit([row for row in rows if row["split"] == val_split_name], args.max_val_samples)
    test_rows = maybe_limit([row for row in rows if row["split"] == "test"], args.max_test_samples)

    print(f"Loaded {len(train_rows)} train / {len(val_rows)} {val_split_name} / {len(test_rows)} test rows")
    print(f"Num labels: {num_labels}")
    print(f"Target labels: {label_names}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
    )

    train_dataset = TargetDataset(train_rows, tokenizer, args.max_len)
    val_dataset = TargetDataset(val_rows, tokenizer, args.max_len)
    test_dataset = TargetDataset(test_rows, tokenizer, args.max_len)
    collate_fn = lambda batch: collate_batch(batch, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    class_weights = None
    if bool(args.use_class_weights):
        class_weights = build_class_weights(train_rows, num_labels)
        print(f"Using class weights: {class_weights.tolist()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(len(train_loader) * max(int(args.epochs), 1), 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(total_steps // 10, 1),
        num_training_steps=total_steps,
    )

    best_val_macro_f1 = -1.0
    best_state_dict = None
    best_epoch = 0
    epochs_without_improvement = 0
    epoch_logs = []
    for epoch_idx in range(int(args.epochs)):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k not in ("labels", "token_type_ids")}
            outputs = model(**inputs)
            loss = loss_fct(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            if step % 20 == 0 or step == len(train_loader):
                print(
                    f"epoch={epoch_idx + 1} step={step}/{len(train_loader)} "
                    f"train_loss={running_loss / step:.4f}"
                )

        val_metrics = evaluate_model(model, val_loader, device)
        epoch_logs.append(
            {
                "epoch": epoch_idx + 1,
                "train_loss": running_loss / max(len(train_loader), 1),
                "val_metrics": val_metrics,
            }
        )
        print(f"epoch={epoch_idx + 1} val_metrics={val_metrics}")
        if val_metrics["macro_f1"] > best_val_macro_f1 + args.early_stop_min_delta:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch_idx + 1
            epochs_without_improvement = 0
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"epoch={epoch_idx + 1} new_best_macro_f1={best_val_macro_f1:.4f}")
        else:
            epochs_without_improvement += 1
            print(
                f"epoch={epoch_idx + 1} no_improve_count={epochs_without_improvement}/"
                f"{args.early_stop_patience}"
            )
            if epochs_without_improvement >= args.early_stop_patience:
                print(
                    f"Early stopping triggered at epoch={epoch_idx + 1}; "
                    f"best_epoch={best_epoch}, best_val_macro_f1={best_val_macro_f1:.4f}"
                )
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    val_metrics = evaluate_model(model, val_loader, device)
    test_metrics = evaluate_model(model, test_loader, device)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metrics_path = Path(args.output_dir) / "metrics_summary.json"
    payload = {
        "model_name": args.model_name,
        "data_path": args.data_path,
        "train_size": len(train_rows),
        "val_split_name": val_split_name,
        "val_size": len(val_rows),
        "test_size": len(test_rows),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "epoch_logs": epoch_logs,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
