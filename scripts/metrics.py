
import numpy as np
import re
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

import unicodedata

# --- Helper for IOU & F1 (Char-level) ---
def vihatet5_standardized_span_metrics(pred_indices, gold_indices, text):
    """
    Standardized metric calculation matching ViHateT5 paper (Nguyen et al., 2024).
    Calculates metrics at the Character level with NFC normalization and per-sentence Macro F1.
    """
    # 1. Normalize and get length (Paper uses len of normalized text)
    norm_text = unicodedata.normalize('NFC', text)
    n = len(norm_text)
    
    if n == 0:
        return 1.0, 1.0, 1.0 # Acc, Macro F1, Binary F1
    
    # 2. Create Binary Masks
    gold_mask = np.zeros(n, dtype=int)
    pred_mask = np.zeros(n, dtype=int)
    
    for idx in gold_indices:
        if 0 <= idx < n:
            gold_mask[idx] = 1
            
    for idx in pred_indices:
        if 0 <= idx < n:
            pred_mask[idx] = 1
            
    # 3. Calculate Metrics
    # Accuracy
    acc = accuracy_score(gold_mask, pred_mask)
    
    # Macro F1 (The key metric in ViHateT5 paper)
    # They use f1_score with average='macro' for EACH sentence.
    macro_f1 = f1_score(gold_mask, pred_mask, average='macro', zero_division=0)
    
    # Weighted F1 (WF1 in Table 3)
    weighted_f1 = f1_score(gold_mask, pred_mask, average='weighted', zero_division=0)
    
    # Binary F1 (Traditional Span F1 for reference)
    binary_f1 = f1_score(gold_mask, pred_mask, average='binary', zero_division=0)
    
    return acc, macro_f1, weighted_f1, binary_f1

def get_char_indices_from_bio_tags(tags, offsets):
    """
    Converts BIO tags and character offsets into a set of character indices.
    """
    indices = set()
    for tag, (start, end) in zip(tags, offsets):
        if tag in [1, 2]: # B-TOXIC or I-TOXIC
            if start < end:
                for i in range(start, end):
                    indices.add(i)
    return indices

def get_char_indices_from_spans(span_indices):
    """
    Converts list of [start, end] into a set of character indices.
    """
    indices = set()
    for s, e in span_indices:
        for i in range(s, e):
            indices.add(i)
    return indices

def calculate_char_metrics(pred_spans, gold_spans, text):
    """
    Legacy helper. Warning: Uses string matching which is ambiguous for repeated words.
    Use vihatet5_standardized_span_metrics for research results.
    """
    n = len(text)
    if n == 0: return 1.0, 1.0, 1.0
    
    pred_mask = np.zeros(n, dtype=int)
    gold_mask = np.zeros(n, dtype=int)
    
    # String matching leads to misalignment if same word appears twice
    def fill_mask(spans, mask):
        for span in spans:
            if not span: continue
            start = 0
            while True:
                idx = text.find(span, start)
                if idx == -1: break
                mask[idx : idx + len(span)] = 1
                start = idx + 1
                
    fill_mask(gold_spans, gold_mask)
    fill_mask(pred_spans, pred_mask)
            
    intersection = np.sum(pred_mask & gold_mask)
    union = np.sum(pred_mask | gold_mask)
    
    iou = 1.0 if union == 0 else intersection / union
    binary_f1 = f1_score(gold_mask, pred_mask, average='binary', zero_division=0)
    seq_macro_f1 = f1_score(gold_mask, pred_mask, average='macro', zero_division=0)
    
    return iou, binary_f1, seq_macro_f1
        
# --- Helper for Token-based Macro F1 ---
def calculate_token_f1(pred_spans, gold_spans, text, tokenizer):
    # FALLBACK: Manual Offset Computation for Slow Tokenizers (PhoBERT)
    try:
        # Try Fast Tokenizer first
        encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoding['offset_mapping']
    except NotImplementedError:
        # Fallback for Python-based Tokenizer
        tokens = tokenizer.tokenize(text)
        offsets = []
        current_pos = 0
        text_lower = text.lower()
        
        for token in tokens:
            # Simple heuristic
            clean_token = token.replace('@@', '')
            try:
                start = text_lower.find(clean_token.lower(), current_pos)
                if start != -1:
                    end = start + len(clean_token)
                    offsets.append((start, end))
                    current_pos = end 
                else:
                    offsets.append((current_pos, current_pos)) 
            except:
                offsets.append((current_pos, current_pos))
        
    n_tokens = len(offsets)
    if n_tokens == 0: return 1.0 if not gold_spans else 0.0

    gold_tags = np.zeros(n_tokens, dtype=int)
    pred_tags = np.zeros(n_tokens, dtype=int)
    
    # Helper to map list of string spans to token labels (0 or 1)
    def map_span_to_tokens(spans, token_labels):
        if not spans: return
        
        # Iterate over all PREDICTED spans
        for span in spans:
            if not span: continue
            
            # Find ALL occurrences of the span in text
            try:
                matches = [m.start() for m in re.finditer(re.escape(span), text)]
            except:
                start = text.find(span)
                matches = [start] if start != -1 else []
            
            for s_start in matches:
                s_end = s_start + len(span)
                
                # Check overlap with tokens
                for idx, (t_start, t_end) in enumerate(offsets):
                    if t_start == t_end: continue
                    
                    # Intersection logic
                    inter_start = max(t_start, s_start)
                    inter_end = min(t_end, s_end)
                    
                    if inter_end > inter_start:
                        token_labels[idx] = 1
    
    map_span_to_tokens(gold_spans, gold_tags)
    map_span_to_tokens(pred_spans, pred_tags)
    
    # Handling All-Zero case
    if np.sum(gold_tags) == 0 and np.sum(pred_tags) == 0:
        return 1.0 
    
    # Calculate Macro F1
    mf1 = f1_score(gold_tags, pred_tags, average='macro', zero_division=0)
    return mf1

# --- Helper for Syllable-level Metrics (Regex) ---
def calculate_syllable_metrics(pred_spans, gold_spans, text):
    """
    Calculates metrics at the syllable level (Regex split).
    """
    vietnamese_word_pattern = r"[a-zA-Z0-9_ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴÝỶỸửữựỳỵỷỹ]+"
    punct_pattern = r"[^\w\s]"
    pattern = f"{vietnamese_word_pattern}|{punct_pattern}"
    
    syllables = re.findall(pattern, text)
    if not syllables: return 1.0, 1.0 
    
    char_to_syllable = {}
    current_char_idx = 0
    syllable_ranges = []
    
    for idx, syl in enumerate(syllables):
        start = text.find(syl, current_char_idx)
        if start == -1: continue
        end = start + len(syl)
        for i in range(start, end):
            char_to_syllable[i] = idx
        syllable_ranges.append((start, end))
        current_char_idx = end
        
    num_syllables = len(syllables)
    y_true_syl = np.zeros(num_syllables, dtype=int)
    y_pred_syl = np.zeros(num_syllables, dtype=int)
    
    def map_spans(spans, target_array):
        for span in spans:
            if not span: continue
            start_search = 0
            while True:
                idx = text.find(span, start_search)
                if idx == -1: break
                end = idx + len(span)
                
                for s_i, (s_start, s_end) in enumerate(syllable_ranges):
                    overlap_start = max(idx, s_start)
                    overlap_end = min(end, s_end)
                    if overlap_end > overlap_start:
                        target_array[s_i] = 1
                start_search = idx + 1

    map_spans(gold_spans, y_true_syl)
    map_spans(pred_spans, y_pred_syl)
    
    if len(y_true_syl) == 0: return 1.0, 1.0, 1.0
    
    acc = accuracy_score(y_true_syl, y_pred_syl)
    binary_f1 = f1_score(y_true_syl, y_pred_syl, zero_division=0)
    macro_f1 = f1_score(y_true_syl, y_pred_syl, average='macro', zero_division=0)
    
    return acc, binary_f1, macro_f1

# --- Helper for Token-based Metrics (Advanced) ---
def calculate_token_metrics_full(pred_spans, gold_spans, text, tokenizer):
    """
    Calculates Accuracy, Precision, Recall, F1 at Token level.
    """
    try:
        encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoding['offset_mapping']
    except:
        tokens = tokenizer.tokenize(text)
        offsets = []
        cursor = 0
        text_lower = text.lower()
        
        for token in tokens:
            clean_t = token.replace("@@", "")
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            start = cursor
            
            if token == tokenizer.unk_token:
                offsets.append((cursor, cursor+1)) 
                cursor += 1
                continue
                
            match_len = len(clean_t)
            if text_lower[cursor:cursor+match_len] == clean_t.lower():
                end = start + match_len
                offsets.append((start, end))
                cursor = end
            else:
                offsets.append((cursor, cursor)) 
                
    n_tokens = len(offsets)
    if n_tokens == 0: return {}

    gold_tags = np.zeros(n_tokens, dtype=int)
    pred_tags = np.zeros(n_tokens, dtype=int)
    
    def map_span_to_tokens(span_list, tag_array):
        for span in span_list:
            if not span: continue
            start_char = 0
            while True:
                idx = text.find(span, start_char)
                if idx == -1: break
                end_char = idx + len(span)
                for i, (ts, te) in enumerate(offsets):
                    overlap_start = max(idx, ts)
                    overlap_end = min(end_char, te)
                    if overlap_end > overlap_start:
                        tag_array[i] = 1
                start_char = idx + 1
    
    map_span_to_tokens(gold_spans, gold_tags)
    map_span_to_tokens(pred_spans, pred_tags)
    
    acc = accuracy_score(gold_tags, pred_tags)
    p, r, f1, _ = precision_recall_fscore_support(gold_tags, pred_tags, average='binary', zero_division=0)
    mf1 = f1_score(gold_tags, pred_tags, average='macro', zero_division=0)
    wf1 = f1_score(gold_tags, pred_tags, average='weighted', zero_division=0)
    
    return {
        "Token Accuracy": acc,
        "Token Precision": p,
        "Token Recall": r,
        "Token F1": f1,
        "Token mF1": mf1,
        "Token wF1": wf1
    }
