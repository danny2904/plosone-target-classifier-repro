import argparse
import importlib
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class LangConfig:
    model_module: str
    model_class: str
    label_map: Dict[str, int]
    normal_label_id: int
    num_labels: int
    default_model_name: str
    default_data_path: str
    default_output_dir: str
    default_max_len: int


def resolve_repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def get_lang_config(lang: str) -> LangConfig:
    root = resolve_repo_root()
    if lang == "vn":
        return LangConfig(
            model_module="model_multitask",
            model_class="XLMRMultiTask",
            label_map={"safe": 0, "unsafe": 1},
            normal_label_id=0,
            num_labels=2,
            default_model_name="xlm-roberta-base",
            default_data_path="data/vihos_prepared.jsonl",
            default_output_dir=os.path.join(root, "experiments/vietnamese/models/unified_backbone"),
            default_max_len=256,
        )
    if lang == "en":
        return LangConfig(
            model_module="experiments.english.proposed.model_multitask_en",
            model_class="HateXplainMultiTaskBIO",
            label_map={"hatespeech": 0, "normal": 1, "offensive": 2},
            normal_label_id=1,
            num_labels=3,
            default_model_name="roberta-base",
            default_data_path=os.path.join(root, "experiments/english/data/hatexplain_prepared.jsonl"),
            default_output_dir=os.path.join(root, "experiments/english/output_multitask_unified"),
            default_max_len=128,
        )
    raise ValueError(f"Unsupported lang: {lang}")


class UnifiedMultiTaskDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_len: int, split: str, cfg: LangConfig):
        self.data: List[dict] = []
        skipped_lines = 0
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if item.get("split") == split:
                        self.data.append(item)
                except Exception:
                    skipped_lines += 1
        if skipped_lines:
            logging.warning("Skipped %d malformed JSONL lines for split '%s'", skipped_lines, split)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cfg = cfg
        self.split = split
        logging.info("Loaded %d samples for split '%s'", len(self.data), split)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _find_sublist(sublist, mainlist):
        len_sub = len(sublist)
        len_main = len(mainlist)
        for i in range(len_main - len_sub + 1):
            if mainlist[i : i + len_sub] == sublist:
                return i
        return -1

    @staticmethod
    def _simple_words(text: str) -> List[str]:
        # Unicode-friendly tokenization for EN/VN short-utterance detection.
        return re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)

    def __getitem__(self, index):
        item = self.data[index]
        text = str(item.get("comment", ""))
        raw_label = str(item.get("label", ""))
        label_id = self.cfg.label_map.get(raw_label, self.cfg.normal_label_id)

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()
        offsets = encoding["offset_mapping"].squeeze(0)

        # Token labels use BIO-style ids expected by existing model heads:
        # 0=Normal, 1=B-toxic, 2=I-toxic, -100=ignore.
        token_labels = torch.zeros(self.max_len, dtype=torch.long)
        token_labels[attention_mask == 0] = -100
        token_labels[offsets[:, 0] == offsets[:, 1]] = -100  # ignore special tokens

        # Prefer char-span alignment (unsafe_spans_indices), fallback to textual spans.
        unsafe_spans = item.get("unsafe_spans_indices") or []
        if isinstance(unsafe_spans, str):
            try:
                unsafe_spans = json.loads(unsafe_spans) if unsafe_spans else []
            except Exception:
                unsafe_spans = []

        if label_id != self.cfg.normal_label_id and unsafe_spans:
            for i in range(self.max_len):
                if attention_mask[i].item() == 0:
                    continue
                start_char, end_char = int(offsets[i, 0].item()), int(offsets[i, 1].item())
                if start_char == end_char:
                    continue
                tok_label = 0
                for span_start, span_end in unsafe_spans:
                    s, e = int(span_start), int(span_end)
                    if max(start_char, s) < min(end_char, e):
                        tok_label = 1 if start_char == s else 2
                        break
                token_labels[i] = tok_label
        elif label_id != self.cfg.normal_label_id:
            spans = item.get("spans", []) or []
            input_id_list = input_ids.tolist()
            for span in spans:
                if not span:
                    continue
                span_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(span))
                start_idx = self._find_sublist(span_ids, input_id_list)
                if start_idx != -1:
                    token_labels[start_idx] = 1
                    for i in range(start_idx + 1, min(start_idx + len(span_ids), self.max_len)):
                        token_labels[i] = 2

        # Optional hard-case upweight (EN/VN): short toxic utterances often under-detected by stage-1.
        cls_sample_weight = 1.0
        token_sample_weight = 1.0
        if (
            getattr(self, "hardcase_cfg", None) is not None
        ):
            hc = self.hardcase_cfg
            if hc.get("enabled", False):
                words = self._simple_words(text)
                is_short = len(words) <= hc.get("max_words", 5)
                if is_short and label_id in hc.get("target_label_ids", set()):
                    has_trigger = any(tok in text.lower() for tok in hc.get("trigger_lexicon", []))
                    if has_trigger:
                        cls_sample_weight = float(hc.get("boost", 1.0))

        sample_id = str(item.get("id", f"{self.split}:{index}"))
        if getattr(self, "hard_ids_set", None):
            if sample_id in self.hard_ids_set:
                token_sample_weight = float(getattr(self, "hard_id_boost", 1.0))
                if getattr(self, "hard_id_apply_to_cls", False):
                    cls_sample_weight = max(cls_sample_weight, token_sample_weight)

        if getattr(self, "token_weight_from_cls_reweight", False):
            token_sample_weight = max(token_sample_weight, cls_sample_weight)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label_id, dtype=torch.long),
            "token_labels": token_labels,
            "cls_sample_weight": torch.tensor(cls_sample_weight, dtype=torch.float),
            "token_sample_weight": torch.tensor(token_sample_weight, dtype=torch.float),
            "sample_id": sample_id,
            "sample_index": torch.tensor(index, dtype=torch.long),
        }

    hardcase_cfg: Dict[str, object] = None
    hard_ids_set: Optional[Set[str]] = None
    hard_id_boost: float = 1.0
    hard_id_apply_to_cls: bool = False
    token_weight_from_cls_reweight: bool = False


def load_model_class(cfg: LangConfig):
    module = importlib.import_module(cfg.model_module)
    return getattr(module, cfg.model_class)


def load_hard_ids(path: str) -> Set[str]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return {str(x) for x in payload}
    if isinstance(payload, dict):
        if "hard_ids" in payload and isinstance(payload["hard_ids"], list):
            return {str(x) for x in payload["hard_ids"]}
    raise ValueError(f"Unsupported hard ids format in {path}")


def compute_token_class_weights(dataset: UnifiedMultiTaskDataset, num_token_labels: int = 3) -> np.ndarray:
    counts = np.zeros(num_token_labels, dtype=np.float64)
    for i in range(len(dataset)):
        tl = dataset[i]["token_labels"].numpy()
        valid = tl[tl >= 0]
        if valid.size == 0:
            continue
        binc = np.bincount(valid, minlength=num_token_labels)
        counts += binc[:num_token_labels]
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_token_labels * counts)
    return weights.astype(np.float32)


def run_training(args):
    cfg = get_lang_config(args.lang)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Training lang=%s on device=%s", args.lang, device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = UnifiedMultiTaskDataset(args.data_path, tokenizer, args.max_len, "train", cfg)
    val_dataset = UnifiedMultiTaskDataset(args.data_path, tokenizer, args.max_len, "val", cfg)

    if args.lang == "en" and args.en_short_toxic_boost > 1.0:
        trigger_lexicon = [
            "fuck you", "stfu", "idiot", "retard", "dumbass", "asshole", "bitch", "cunt",
            "moron", "piece of shit", "go to hell", "fucking", "trash",
        ]
        hardcase_cfg = {
            "enabled": True,
            "boost": float(args.en_short_toxic_boost),
            "max_words": int(args.en_short_max_words),
            "target_label_ids": {0, 2},  # hate/offensive
            "trigger_lexicon": trigger_lexicon,
        }
        train_dataset.hardcase_cfg = hardcase_cfg
        logging.info(
            "EN hard-case reweight enabled: boost=%.2f, max_words=%d, lexicon_size=%d",
            args.en_short_toxic_boost,
            args.en_short_max_words,
            len(trigger_lexicon),
        )
    elif args.lang == "vn" and args.vn_short_unsafe_boost > 1.0:
        trigger_lexicon = [
            "đồ ngu", "ngu", "ngu học", "óc chó", "đần", "khốn nạn", "vô học", "con chó",
            "cút", "biến đi", "địt", "đm", "dm", "cc", "vl", "vkl",
        ]
        hardcase_cfg = {
            "enabled": True,
            "boost": float(args.vn_short_unsafe_boost),
            "max_words": int(args.vn_short_max_words),
            "target_label_ids": {1},  # unsafe
            "trigger_lexicon": trigger_lexicon,
        }
        train_dataset.hardcase_cfg = hardcase_cfg
        logging.info(
            "VN hard-case reweight enabled: boost=%.2f, max_words=%d, lexicon_size=%d",
            args.vn_short_unsafe_boost,
            args.vn_short_max_words,
            len(trigger_lexicon),
        )

    if args.hard_ids_path:
        hard_ids = load_hard_ids(args.hard_ids_path)
        train_dataset.hard_ids_set = hard_ids
        train_dataset.hard_id_boost = float(args.hard_id_boost)
        train_dataset.hard_id_apply_to_cls = bool(args.hard_id_apply_to_cls)
        logging.info(
            "Loaded %d hard ids from %s (boost=%.2f, apply_to_cls=%s)",
            len(hard_ids),
            args.hard_ids_path,
            args.hard_id_boost,
            bool(args.hard_id_apply_to_cls),
        )

    train_dataset.token_weight_from_cls_reweight = bool(args.token_weight_from_cls_reweight)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model_cls = load_model_class(cfg)
    model = model_cls.from_pretrained(args.model_name, num_labels=cfg.num_labels, use_fusion=args.use_fusion)
    if args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location="cpu")
        if args.lang == "en":
            # Robust remap for legacy EN checkpoints saved with older key conventions.
            remap = {}
            for k, v in state.items():
                new_key = k
                if k.startswith("bert."):
                    new_key = k.replace("bert.", "roberta.")
                elif not k.startswith("roberta.") and not any(x in k for x in ["classifier", "token_"]):
                    new_key = "roberta." + k
                if "class_classifier" in new_key:
                    new_key = new_key.replace("class_classifier", "cls_classifier")
                if new_key.startswith("roberta.cls_classifier"):
                    new_key = new_key.replace("roberta.", "")
                if new_key.startswith("roberta.token_classifier"):
                    new_key = new_key.replace("roberta.", "")
                remap[new_key] = v

            if "token_classifier.weight" in remap:
                ckpt_size = remap["token_classifier.weight"].shape[0]
                cur_size = model.token_classifier.weight.shape[0]
                if ckpt_size != cur_size:
                    logging.warning(
                        "Init checkpoint token head has %d labels, model expects %d. "
                        "Skipping legacy token head weights.",
                        ckpt_size,
                        cur_size,
                    )
                    remap.pop("token_classifier.weight", None)
                    remap.pop("token_classifier.bias", None)

            load_info = model.load_state_dict(remap, strict=False)
            missing_backbone = [k for k in load_info.missing_keys if k.startswith("roberta.")]
            if missing_backbone:
                raise RuntimeError(
                    f"Failed to load EN init checkpoint backbone; missing roberta keys: {missing_backbone[:8]}"
                )
            logging.info(
                "Loaded EN init checkpoint with remap from %s (missing=%d unexpected=%d)",
                args.init_checkpoint,
                len(load_info.missing_keys),
                len(load_info.unexpected_keys),
            )
        else:
            model.load_state_dict(state, strict=True)
            logging.info("Loaded init checkpoint from %s", args.init_checkpoint)
    model.to(device)

    cls_class_weights = None
    if args.use_class_weights:
        labels_arr = np.array(
            [cfg.label_map.get(str(x.get("label", "")), cfg.normal_label_id) for x in train_dataset.data],
            dtype=np.int64,
        )
        counts = np.bincount(labels_arr, minlength=cfg.num_labels).astype(np.float32)
        counts = np.maximum(counts, 1.0)
        weights = counts.sum() / (cfg.num_labels * counts)
        cls_class_weights = torch.tensor(weights, dtype=torch.float, device=device)
        logging.info("Using class weights: %s", weights.tolist())

    token_class_weights = None
    if args.use_token_class_weights:
        t_weights = compute_token_class_weights(train_dataset, num_token_labels=3)
        token_class_weights = torch.tensor(t_weights, dtype=torch.float, device=device)
        logging.info("Using token class weights: %s", t_weights.tolist())

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float("inf")
    best_f1 = -1.0
    best_acc = -1.0
    best_token_f1 = -1.0
    epochs_no_improve = 0

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [], "val_token_f1": []}

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            token_labels = batch["token_labels"].to(device)
            cls_sample_weights = batch.get("cls_sample_weight")
            if cls_sample_weights is not None:
                cls_sample_weights = cls_sample_weights.to(device)
            token_sample_weights = batch.get("token_sample_weight")
            if token_sample_weights is not None:
                token_sample_weights = token_sample_weights.to(device)

            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                token_labels=token_labels,
                alpha=args.alpha,
                use_consistency=args.use_consistency,
                lambda_const=args.lambda_const,
                cls_class_weights=cls_class_weights,
                cls_sample_weights=cls_sample_weights,
                token_class_weights=token_class_weights,
                token_focal_gamma=args.token_focal_gamma,
                token_dice_lambda=args.token_dice_lambda,
                token_sample_weights=token_sample_weights,
            )
            loss = outputs["loss"]
            if loss is None:
                raise RuntimeError("Model returned None loss during training")
            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss at epoch={epoch+1}, step={step}")

            total_train_loss += float(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / max(1, len(train_loader))

        model.eval()
        total_val_loss = 0.0
        all_preds = []
        all_labels = []
        all_tok_preds = []
        all_tok_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                token_labels = batch["token_labels"].to(device)
                cls_sample_weights = batch.get("cls_sample_weight")
                if cls_sample_weights is not None:
                    cls_sample_weights = cls_sample_weights.to(device)
                token_sample_weights = batch.get("token_sample_weight")
                if token_sample_weights is not None:
                    token_sample_weights = token_sample_weights.to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    token_labels=token_labels,
                    alpha=args.alpha,
                    use_consistency=args.use_consistency,
                    lambda_const=args.lambda_const,
                    cls_class_weights=cls_class_weights,
                    cls_sample_weights=cls_sample_weights,
                    token_class_weights=token_class_weights,
                    token_focal_gamma=args.token_focal_gamma,
                    token_dice_lambda=args.token_dice_lambda,
                    token_sample_weights=token_sample_weights,
                )
                total_val_loss += float(outputs["loss"].item())
                preds = torch.argmax(outputs["cls_logits"], dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                tok_preds = torch.argmax(outputs["token_logits"], dim=-1).cpu().numpy()
                tok_labels = token_labels.cpu().numpy()
                valid = tok_labels != -100
                if valid.any():
                    all_tok_preds.extend((tok_preds[valid] > 0).astype(np.int32).tolist())
                    all_tok_labels.extend((tok_labels[valid] > 0).astype(np.int32).tolist())

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        val_token_f1 = f1_score(all_tok_labels, all_tok_preds, zero_division=0) if all_tok_labels else 0.0

        logging.info(
            "Epoch %d: train_loss=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f val_token_f1=%.4f",
            epoch + 1,
            avg_train_loss,
            avg_val_loss,
            val_acc,
            val_f1,
            val_token_f1,
        )

        improved = False
        if args.best_metric == "val_f1":
            improved = val_f1 > best_f1
        elif args.best_metric == "val_token_f1":
            improved = val_token_f1 > best_token_f1
        elif args.best_metric == "val_acc":
            improved = val_acc > best_acc
        else:
            improved = avg_val_loss < best_loss

        if improved:
            best_loss = min(best_loss, avg_val_loss)
            best_f1 = max(best_f1, val_f1)
            best_acc = max(best_acc, val_acc)
            best_token_f1 = max(best_token_f1, val_token_f1)
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            with open(os.path.join(args.output_dir, "best_checkpoint_meta.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "lang": args.lang,
                        "epoch": epoch + 1,
                        "best_metric": args.best_metric,
                        "val_loss": avg_val_loss,
                        "val_f1_macro": val_f1,
                        "val_token_f1": val_token_f1,
                        "val_acc": val_acc,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            logging.info("Saved new best checkpoint.")
        else:
            epochs_no_improve += 1
            if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
                logging.info(
                    "Early stopping at epoch %d: no improvement for %d epoch(s).",
                    epoch + 1,
                    epochs_no_improve,
                )
                history["train_loss"].append(avg_train_loss)
                history["val_loss"].append(avg_val_loss)
                history["val_acc"].append(val_acc)
                history["val_f1"].append(val_f1)
                history["val_token_f1"].append(val_token_f1)
                break

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_token_f1"].append(val_token_f1)

    with open(os.path.join(args.output_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    logging.info("Training complete. Output: %s", args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified EN/VN multitask backbone trainer.")
    parser.add_argument("--lang", type=str, required=True, choices=["en", "vn"])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=5.0, help="Token loss weight.")
    parser.add_argument("--use_fusion", type=int, default=1)
    parser.add_argument("--use_consistency", type=int, default=1)
    parser.add_argument("--lambda_const", type=float, default=0.1)
    parser.add_argument("--use_class_weights", type=int, default=1)
    parser.add_argument("--best_metric", type=str, default="val_f1", choices=["val_f1", "val_token_f1", "val_acc", "val_loss"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init_checkpoint", type=str, default=None, help="Path to .pth state_dict to continue training.")
    parser.add_argument("--early_stop_patience", type=int, default=0, help="Stop if no best-metric improvement for N consecutive epochs.")
    parser.add_argument(
        "--en_short_toxic_boost",
        type=float,
        default=1.0,
        help="EN only: upweight short toxic/offensive training samples (1.0 disables).",
    )
    parser.add_argument(
        "--en_short_max_words",
        type=int,
        default=5,
        help="EN only: maximum word count considered short utterance for hard-case boost.",
    )
    parser.add_argument(
        "--vn_short_unsafe_boost",
        type=float,
        default=1.0,
        help="VN only: upweight short unsafe training samples (1.0 disables).",
    )
    parser.add_argument(
        "--vn_short_max_words",
        type=int,
        default=6,
        help="VN only: maximum word count considered short utterance for hard-case boost.",
    )
    parser.add_argument(
        "--use_token_class_weights",
        type=int,
        default=0,
        help="Enable token-level class weights (for O/B/I imbalance).",
    )
    parser.add_argument(
        "--token_focal_gamma",
        type=float,
        default=0.0,
        help="Token focal gamma (>0 enables focal modulation on token CE).",
    )
    parser.add_argument(
        "--token_dice_lambda",
        type=float,
        default=0.0,
        help="Weight for toxic-mask dice loss regularizer.",
    )
    parser.add_argument(
        "--hard_ids_path",
        type=str,
        default=None,
        help="Optional JSON file containing hard sample ids to upweight token loss.",
    )
    parser.add_argument(
        "--hard_id_boost",
        type=float,
        default=1.0,
        help="Token loss sample-weight boost for ids in hard_ids_path.",
    )
    parser.add_argument(
        "--hard_id_apply_to_cls",
        type=int,
        default=0,
        help="Also apply hard-id boost to cls loss (default: token-only).",
    )
    parser.add_argument(
        "--token_weight_from_cls_reweight",
        type=int,
        default=0,
        help="Mirror short-toxic cls reweight into token sample weights.",
    )
    args = parser.parse_args()

    cfg = get_lang_config(args.lang)
    if args.data_path is None:
        args.data_path = cfg.default_data_path
    if args.model_name is None:
        args.model_name = cfg.default_model_name
    if args.output_dir is None:
        args.output_dir = cfg.default_output_dir
    if args.max_len is None:
        args.max_len = cfg.default_max_len

    args.use_fusion = bool(args.use_fusion)
    args.use_consistency = bool(args.use_consistency)
    args.use_class_weights = bool(args.use_class_weights)
    args.use_token_class_weights = bool(args.use_token_class_weights)
    args.hard_id_apply_to_cls = bool(args.hard_id_apply_to_cls)
    args.token_weight_from_cls_reweight = bool(args.token_weight_from_cls_reweight)
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    run_training(args)


if __name__ == "__main__":
    main()
