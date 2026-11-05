import os
import sys
import torch
import json
import numpy as np
import argparse
import time
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from transformers import AutoTokenizer
try:
    import google.generativeai as genai
except ModuleNotFoundError:
    genai = None
from torch.utils.data import DataLoader, Dataset
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False

# Load env variables from .env file if present
load_dotenv()

# Add path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from config_loader import MODEL_PATH, DATA_PATH
from model_multitask import XLMRMultiTask as PhoBERTMultiTask

# --- IMPORT COMMON METRICS & REPORTING ---
# package layout: metrics.py and reporting.py sit next to this file
common_dir = current_dir
if common_dir not in sys.path: sys.path.append(common_dir)

from metrics import (
    vihatet5_standardized_span_metrics, 
    get_char_indices_from_bio_tags,
    get_char_indices_from_spans,
    calculate_syllable_metrics,
    calculate_token_metrics_full
)
from reporting import generate_report_string
import ast

# --- Context / Knowledge Knowledge Base (Teencode/Slang) ---
VIETNAMESE_SLANG_KB = """
LƯU Ý ĐẶC BIỆT VỀ TỪ LÓNG VIỆT NAM (TEENCODE):
1. 'má', 'mẹ', 'bà': 
   - Nếu dùng để gọi người thân hoặc cảm thán nhẹ ("má ơi", "mẹ nó", "bà già"): KHÔNG PHẢI HATE (SAFE).
   - Nếu chửi thề trực tiếp ("đ* má", "con mẹ mày", "mẹ kiếp"): LÀ HATE (TOXIC).
2. 'đồ', 'hàng':
   - Nếu chỉ vật dụng ("đồ chơi", "hàng hóa"): SAFE.
   - Nếu miệt thị ("đồ ngu", "con hàng", "đồ chó"): TOXIC.
3. 'đồng bóng', 'pđ', 'bede', 'bóng cái': Từ lóng miệt thị giới tính (LGBT): TOXIC (HATE).
4. 'bug', 'lag', 'hack': Thuật ngữ game/công nghệ: SAFE.
5. 'phò', 'cave', 'đi khách': Miệt thị phụ nữ hoặc nghề nghiệp nhạy cảm: TOXIC.
6. 'bắc kỳ', 'nam kỳ', 'ba que', 'bò đỏ': Từ phân biệt vùng miền/chính trị cực đoan: TOXIC.
"""

# --- METRIC FUNCTIONS REMOVED (Now imported from common/metrics.py) ---

class InferenceDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=256, split='test', limit=None):
        self.data = []
        if data_path.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(data_path)
            if 'source' in df.columns:
                df = df[df['source'].str.contains("hate-spans-detection")].copy()
            self.data = df.to_dict('records')
        else:
            try:
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        if split is None or item.get('split') == split:
                            self.data.append(item)
            except FileNotFoundError:
                print(f"File not found: {data_path}")
        
        if limit:
            self.data = self.data[:limit]
            
        self.tokenizer = tokenizer
        self.max_len = max_len
        print(f"Loaded {len(self.data)} samples for split '{split}'")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        text = str(item.get('comment', ''))
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'text': text
        }

# --- RATeD Model Wrapper ---
class RATeDInference:
    def __init__(self, model_path, model_name="vinai/phobert-base-v2", use_fusion=True, device=None):
        # Default to cuda:1 if available and 2 GPUs exist, else cuda:0
        if not device and torch.cuda.device_count() > 1:
            self.device = torch.device('cuda:1')
        else:
            self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading RATeD model from {model_path} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=True)
        self.model = PhoBERTMultiTask.from_pretrained(model_name, num_labels=2, use_fusion=use_fusion)
        
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading model weights: {e}")
            raise e
            
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, texts, threshold_cls=0.5):
        """
        Batched prediction for RATeD (XLM-R).
        """
        if isinstance(texts, str): texts = [texts]
        
        encodings = self.tokenizer(
            texts,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=encodings['input_ids'], attention_mask=encodings['attention_mask'])
            
            # 1. Classification logic for batch
            cls_probs_all = torch.softmax(outputs['cls_logits'], dim=1).cpu().numpy()
            token_preds_all = torch.argmax(outputs['token_logits'], dim=2).cpu().numpy()
            offsets_all = encodings['offset_mapping'].cpu().numpy()
            
            batch_results = []
            for i, text in enumerate(texts):
                probs = cls_probs_all[i]
                toxic_prob = probs[1]
                is_toxic = toxic_prob > threshold_cls
                preds = token_preds_all[i]
                offsets = offsets_all[i]
                
                pred_indices = set()
                for p, (st, en) in zip(preds, offsets):
                    if st == en: continue
                    if p in [1, 2]: # B or I
                        for idx in range(st, en): pred_indices.add(idx)
                
                spans = self._decode_bio_spans(text, preds, offsets)
                
                batch_results.append({
                    'is_toxic': bool(is_toxic),
                    'confidence': float(toxic_prob),
                    'spans': spans,
                    'char_indices': list(pred_indices),
                    'token_logits': outputs['token_logits'][i]
                })
                
        return batch_results

    def _decode_bio_spans(self, text, preds, offsets):
        spans = []
        current_start, current_end = -1, -1
        
        for p, (st, en) in zip(preds, offsets):
            if st == en: continue
            if p == 1: # B-Toxic
                if current_start != -1:
                    spans.append(text[current_start:current_end])
                current_start, current_end = st, en
            elif p == 2: # I-Toxic
                if current_start != -1:
                    current_end = max(current_end, en)
                else: # Fallback: Treat orphan I as B
                    current_start, current_end = st, en
            else: # O
                if current_start != -1:
                    spans.append(text[current_start:current_end])
                    current_start, current_end = -1, -1
                    
        if current_start != -1:
            spans.append(text[current_start:current_end])
            
        return [s.strip() for s in spans if s.strip()]

# --- LLM Judge Class ---
class LLMJudge:
    def __init__(self, api_key=None, max_workers=8, no_slang=False, bottleneck=False):
        self.no_slang = no_slang
        self.bottleneck = bottleneck
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")
        self.max_workers = max_workers
        if not api_key:
            print("WARNING: No GOOGLE_API_KEY found. LLM Judge will fail.")
            self.model = None
        else:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
            print(f"LLM Judge (Gemini 2.5 Flash Lite) initialized (parallel workers={max_workers}).")

        if self.bottleneck:
            # Simplified "Vanilla" Prompt (No rigorous definition, no examples)
            # This simulates a standard user query without prompt engineering
            self.instruction = "Bạn là hệ thống phân tích toxicity. Hãy trích xuất các cụm từ độc hại trong câu sau. Nếu sạch, trả về 'SAFE'."
            self.max_output_tokens = 50 # Relaxed from 8 to allow full spans
        else:
            slang_kb = "" if self.no_slang else VIETNAMESE_SLANG_KB
            self.instruction = f"""Bạn là một chuyên gia phân tích ngôn ngữ độc hại (Toxic Detection), chuyên hỗ trợ kiểm duyệt nội dung trên mạng xã hội Việt Nam.
Nhiệm vụ của bạn là xác định văn bản có chứa các nội dung sau không: Ngôn từ thù ghét (Hate speech), Xúc phạm (Offensive), Tục tĩu (Profanity), hoặc Quấy rối (Harassment).

{slang_kb}

Chỉ trả về 'SAFE' hoặc các cụm từ độc hại cách nhau bởi dấu phẩy. Không giải thích gì thêm."""
            self.max_output_tokens = 50

    def _call_one(self, text, span=""):
        if self.bottleneck:
            prompt = f"""
            {self.instruction}

            Ngữ cảnh: "{text}"
            """
        else:
            prompt = f"""
            {self.instruction}

            Ngữ cảnh: "{text}"
            Cụm từ nghi ngờ: "{span}"
            """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.0)
            )
            return (response.text or "").strip()
        except Exception as e:
            print(f"LLM Error: {e}", flush=True)
            return "SAFE"

    def verify_batch(self, texts):
        """Gọi Gemini song song cho nhiều văn bản — giảm thời gian chạy rất nhiều."""
        if not texts:
            return []
        if not self.model:
            return ["SAFE"] * len(texts)
        n = len(texts)
        res = [None] * n
        workers = min(self.max_workers, n)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._call_one, t): i for i, t in enumerate(texts)}
            for fut in as_completed(futures):
                idx = futures[fut]
                raw = (fut.result() or "").strip()
                if "SAFE" in raw.upper() or "KHÔNG" in raw.upper():
                    res[idx] = "SAFE"
                else:
                    res[idx] = raw.strip(' ".\'')
        return res

    def verify_span(self, text, span):
        if not self.model:
            return span
        raw = self._call_one(text)
        if "SAFE" in raw.upper() or "KHÔNG" in raw.upper():
            return None
        refined_span = raw.strip(' ".\'')
        if refined_span and refined_span not in text:
            return span
        return refined_span if refined_span else span

class HFJudge:
    def __init__(self, api_key=None, model_repo="Qwen/Qwen2.5-7B-Instruct"):
        from huggingface_hub import InferenceClient
        if not api_key:
            api_key = os.environ.get("HF_TOKEN")
        
        if not api_key:
            print("WARNING: No HF_TOKEN found. HF Judge will fail.")
            self.client = None
        else:
            self.client = InferenceClient(token=api_key)
            self.model_repo = model_repo
            print(f"HF Judge initialized with model: {model_repo}")

    def verify_span(self, text, span):
        if not self.client:
            return True # Safety fallback
            
        prompt = f"""
        Bạn là một chuyên gia ngôn ngữ học và kiểm duyệt nội dung.
        
        Ngữ cảnh: "{text}"
        Cụm từ bị nghi ngờ: "{span}"
        
        Câu hỏi: Cụm từ "{span}" trong ngữ cảnh trên CÓ PHẢI là ngôn từ thù ghét, xúc phạm, hoặc độc hại không?
        Lưu ý:
        - Nếu đó là từ lóng mang nghĩa xúc phạm: Trả lời CÓ.
        - Nếu đó là từ bình thường bị bắt nhầm: Trả lời KHÔNG.
        - Nếu đó là nhận xét tiêu cực nhưng không thô tục/thù ghét: Trả lời KHÔNG.
        
        Chỉ trả lời đúng 1 từ: "CÓ" hoặc "KHÔNG".
        """
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Mistral/Qwen/Gemma usually standard Chat completion
            response = self.client.chat_completion(
                model=self.model_repo,
                messages=messages,
                max_tokens=10,
                temperature=0.0 # FORCE DETERMINISTIC
            )
            answer = response.choices[0].message.content.strip().upper()
            if "CÓ" in answer or "YES" in answer:
                return span # Toxic confirmed -> Return original string
            return None # Clean -> Return None
        except Exception as e:
            return span # Fallback: Assume Toxic to be safe

class LocalJudge:
    def __init__(self, model_repo="Qwen/Qwen2.5-7B-Instruct", load_in_4bit=True, no_slang=False, bottleneck=False):
        self.no_slang = no_slang
        self.bottleneck = bottleneck
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
        import torch
        import os
        import json

        print(f"Loading local model: {model_repo} (Offline Inference)...")
        
        # Explicit CUDA Check
        if torch.cuda.is_available():
            print(f"✅ CUDA detected. Available GPUs: {torch.cuda.device_count()}")
            print(f"Current GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("❌ CUDA NOT DETECTED! Local inference will be extremely slow on CPU.")

        # Check for LoRA Adapter
        is_adapter = os.path.exists(os.path.join(model_repo, "adapter_config.json"))
        
        base_model_name = model_repo
        adapter_path = None
        
        if is_adapter:
            print(f"⚠️ DETECTED LORA ADAPTER at {model_repo}")
            adapter_path = model_repo
            # Read base model from config
            with open(os.path.join(model_repo, "adapter_config.json"), 'r') as f:
                adapter_conf = json.load(f)
            base_model_orig = adapter_conf.get("base_model_name_or_path", "Qwen/Qwen2.5-7B-Instruct")
            
            # Map unsloth base to official HF repo if needed (to avoid unsloth dependency if not installed)
            if "unsloth" in base_model_orig and "qwen2.5-7b-instruct" in base_model_orig.lower():
                print(f"Mapping unsloth base '{base_model_orig}' to 'Qwen/Qwen2.5-7B-Instruct' for transformers compatibility.")
                base_model_name = "Qwen/Qwen2.5-7B-Instruct" 
            else:
                base_model_name = base_model_orig
                
            print(f"-> Base Model inferred: {base_model_name}")
        
        is_local_base = os.path.exists(base_model_name)
        
        # Load Tokenizer (Use adapter's tokenizer if available, else base)
        tokenizer_path = adapter_path if adapter_path and os.path.exists(os.path.join(adapter_path, "tokenizer.json")) else base_model_name
        print(f"Loading tokenizer from: {tokenizer_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, 
            trust_remote_code=True,
            local_files_only=os.path.exists(tokenizer_path)
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left' # Required for batch generation
            
        # Quantization Config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ) if load_in_4bit else None
        
        print(f"Loading Base Model: {base_model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name, 
            device_map="auto", 
            quantization_config=bnb_config,
            # local_files_only=is_local_base, # Configurable?
            trust_remote_code=True,
            torch_dtype=torch.float16 
        )

        if is_adapter:
            from peft import PeftModel
            print(f"Loading LoRA Adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            print("✅ LoRA Adapter merged/loaded successfully.")
        
        # PROOF OF EXECUTION
        print(f"✅ MODEL IS DISTRIBUTED ACROSS: {self.model.hf_device_map}")
        
        print(f"Model loaded. Memory footprint: {self.model.get_memory_footprint() / 1e6:.2f} MB")
        
        # self.pipe is not strictly needed if we use self.model.generate directly, but kept for compatibility
        # self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        print("Local Judge initialized successfully.")

    def verify_span(self, text, span=""):
        # Wrapper for single text using batch logic
        results = self.verify_batch([text])
        return results[0] if results else None

    def verify_batch(self, texts):
        if not texts: return []
        
        instruction = "Bạn là chuyên gia phân tích ngôn từ thù ghét trên mạng xã hội Việt Nam. Hãy trích xuất chính xác tất cả các từ hoặc cụm từ độc hại, xúc phạm hoặc thù ghét trong văn bản. Nếu văn bản an toàn, chỉ trả về 'SAFE'."
        
        formatted_inputs = []
        for text in texts:
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": text}
            ]
            text_input = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted_inputs.append(text_input)
            
        try:
            # Native model.generate is more robust for batching
            inputs = self.tokenizer(formatted_inputs, return_tensors="pt", padding=True).to(self.model.device)
            
            # Fix: Define max_new_tokens explicitly
            max_new_tokens = 100 
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Extract only the newly generated tokens
            input_len = inputs.input_ids.shape[1]
            generated_tokens = outputs[:, input_len:]
            
            answers = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            results = []
            for answer in answers:
                ans = answer.strip()
                results.append("SAFE" if "SAFE" in ans.upper() else ans)
            return results
            
        except Exception as e:
            # FIX: Print to sys.stderr or use a way to ensure it's seen. 
            # Since we don't have the logger object here, print with flush=True is best effort
            print(f"\n[CRITICAL ERROR] Local Model Batch Error: {e}", flush=True) 
            # traceback.print_exc() # detailed trace
            return [None] * len(texts)

class LocalPhoBERTJudge:
    """Stage-2 specialist judge using PhoBERT classifier and optional span extractor."""
    def __init__(self, cls_model_repo="joshswift/phobert-toxicity", span_model_repo="joshswift/phobert-span", cls_threshold=0.5):
        from transformers import pipeline
        import torch

        self.cls_threshold = cls_threshold
        device = 0 if torch.cuda.is_available() else -1

        print(f"Loading PhoBERT classifier judge: {cls_model_repo}")
        self.cls_pipe = pipeline(
            "text-classification",
            model=cls_model_repo,
            tokenizer=cls_model_repo,
            device=device
        )

        self.span_pipe = None
        if span_model_repo:
            try:
                print(f"Loading PhoBERT span judge: {span_model_repo}")
                self.span_pipe = pipeline(
                    "token-classification",
                    model=span_model_repo,
                    tokenizer=span_model_repo,
                    aggregation_strategy="simple",
                    device=device
                )
            except Exception as e:
                print(f"WARNING: Failed to load span model ({span_model_repo}): {e}")
                self.span_pipe = None

        print("Local PhoBERT Judge initialized successfully.")

    @staticmethod
    def _clean_span_text(text):
        if not text:
            return ""
        cleaned = text.replace("▁", " ").replace("_", " ").strip()
        return " ".join(cleaned.split())

    def verify_span(self, text, span=""):
        results = self.verify_batch([text])
        return results[0] if results else None

    def verify_batch(self, texts):
        if not texts:
            return []

        cls_outputs = self.cls_pipe(texts, truncation=True, max_length=256, batch_size=32)
        final = []

        for text, pred in zip(texts, cls_outputs):
            label = str(pred.get("label", "")).lower()
            score = float(pred.get("score", 0.0))
            is_unsafe = (("unsafe" in label) or ("label_1" in label)) and (score >= self.cls_threshold)

            if not is_unsafe:
                final.append("SAFE")
                continue

            if self.span_pipe is None:
                final.append(text)
                continue

            try:
                spans_raw = self.span_pipe(text, truncation=True, max_length=256)
                spans = []
                seen = set()
                for item in spans_raw:
                    token = self._clean_span_text(item.get("word", ""))
                    if token and token not in seen:
                        seen.add(token)
                        spans.append(token)
                final.append(", ".join(spans) if spans else text)
            except Exception:
                final.append(text)

        return final

# --- Cascaded Pipeline ---
class CascadedPipeline:
    def __init__(self, rated_model, llm_judge, only_stage1=False, only_stage2=False):
        self.rated = rated_model
        self.judge = llm_judge
        self.only_stage1 = only_stage1
        self.only_stage2 = only_stage2
        self.stats = {
            "total": 0,
            "rated_clean": 0,
            "rated_verify": 0,
            "rated_high_conf_bypass": 0, 
            "confirmed_toxic": 0,
            "judge_flipped_to_clean": 0,
            "judge_confirmed_clean": 0
        }

    def process(self, text):
        # Wrapper for single text inference
        return self.process_batch([text])[0]

    def process_batch(self, texts):
        batch_size = len(texts)
        self.stats["total"] += batch_size
        
        if self.only_stage2:
            # ONLY STAGE 2: Bypass backbone, send ALL to Judge
            to_verify_indices = list(range(batch_size))
            to_verify_texts = texts
            # Dummy results
            rated_results = [{"is_toxic": False, "confidence": 0.0, "spans": [], "char_indices": []} for _ in range(batch_size)]
            final_results = [None] * batch_size
        else:
            # 1. RATeD Inference
            rated_results = self.rated.predict(texts)
            final_results = [None] * batch_size
            to_verify_indices = []
            to_verify_texts = []

        if not self.only_stage2:
            for i, (text, r_result) in enumerate(zip(texts, rated_results)):
                is_toxic_rated = r_result['is_toxic']
                confidence = r_result['confidence']
                
                # BRANCHING
                if self.only_stage1:
                    is_judge_needed = False
                else:
                    # Logic for Judge verification
                    is_judge_needed = not (
                        (is_toxic_rated and confidence > 0.90) or 
                        (not is_toxic_rated and confidence >= 0.98)
                    )

                if not is_judge_needed:
                    if is_toxic_rated:
                        self.stats["rated_high_conf_bypass"] += 1
                        label = "hate"
                    else:
                        self.stats["rated_clean"] += 1
                        label = "clean"
                        
                    final_results[i] = {
                        "label": label,
                        "spans": r_result['spans'],
                        "char_indices": r_result['char_indices'],
                        "flow": "FAST_PATH_" + label.upper(),
                        "original_rated": r_result
                    }
                else:
                    to_verify_indices.append(i)
                    to_verify_texts.append(text)
        
        # 2. Judge Verification in Batch
        if to_verify_indices:
            self.stats["rated_verify"] += len(to_verify_indices)
            
            if hasattr(self.judge, "verify_batch"):
                judge_results = self.judge.verify_batch(to_verify_texts)
            else:
                judge_results = [self.judge.verify_span(t, t[:100]) for t in to_verify_texts]
                
            for idx, text_idx in enumerate(to_verify_indices):
                text = texts[text_idx]
                r_result = rated_results[text_idx]
                judge_refined = judge_results[idx]
                
                is_toxic_rated = r_result['is_toxic']
                
                # Logic for Judge decisions
                if judge_refined and "SAFE" in judge_refined.upper():
                    # Judge says clean
                    if is_toxic_rated: self.stats["judge_flipped_to_clean"] += 1
                    else: self.stats["judge_confirmed_clean"] += 1
                    
                    final_results[text_idx] = {
                        "label": "clean",
                        "spans": [],
                        "char_indices": [],
                        "flow": "JUDGE_CLEAN",
                        "original_rated": r_result
                    }
                else:
                    # Judge says toxic
                    if not is_toxic_rated: self.stats["confirmed_toxic"] += 1
                    
                    final_indices = set(r_result['char_indices'])
                    final_spans = r_result['spans']
                    
                    if judge_refined and judge_refined != "SAFE":
                        j_spans = [s.strip() for s in judge_refined.split(",") if s.strip()]
                        for s in j_spans:
                            start = 0
                            while True:
                                found_idx = text.find(s, start)
                                if found_idx == -1: break
                                for char_idx in range(found_idx, found_idx + len(s)): final_indices.add(char_idx)
                                start = found_idx + 1
                            if s not in final_spans: final_spans.append(s)
                            
                    final_results[text_idx] = {
                        "label": "hate",
                        "spans": final_spans,
                        "char_indices": list(final_indices),
                        "flow": "JUDGE_TOXIC",
                        "original_rated": r_result
                    }
                    
        return final_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--output", type=str, default="cascaded_results.json")
    parser.add_argument("--provider", type=str, default="local", choices=["gemini", "hf", "local", "local_cls"], help="LLM Provider")
    parser.add_argument("--model_path", type=str, default=None, help="Backbone checkpoint path (.pth).")
    parser.add_argument("--hf_model", type=str, default="experiments/vietnamese/models/qwen2.5-7b-vihos-specialist", help="HF Repo or Local Path")
    parser.add_argument("--cls_model", type=str, default="joshswift/phobert-toxicity", help="Classifier specialist model (for local_cls)")
    parser.add_argument("--span_model", type=str, default="joshswift/phobert-span", help="Span specialist model (for local_cls)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for faster inference")
    parser.add_argument("--load_8bit", action="store_true", help="Load local model in 8bit/4bit")
    parser.add_argument("--only_stage1", action="store_true", help="Bypass Judge, evaluate only RATeD backbone.")
    parser.add_argument("--only_stage2", action="store_true", help="Bypass Backbone, evaluate only LLM Judge.")
    parser.add_argument("--no_slang", action="store_true", help="Disable Slang KB in LLM Prompts.")
    parser.add_argument("--bottleneck", action="store_true", help="Cripple LLM performance for baseline study (Vanilla prompt + Short output).")
    parser.add_argument("--use_fusion", type=int, default=1, help="Whether the checkpoint was trained with rationale-guided fusion (must match training config).")
    args = parser.parse_args()

    # Init
    from config_loader import MODEL_DIR
    model_id = "xlmr_multitask_bio_v4"
    best_model_path = args.model_path or os.path.join(MODEL_DIR, model_id, "best_multitask_model.pth")
    rated = RATeDInference(best_model_path, model_name="xlm-roberta-base", use_fusion=bool(args.use_fusion))
    
    if args.only_stage1:
        judge = None
    elif args.provider == "gemini":
        judge = LLMJudge(no_slang=args.no_slang, bottleneck=args.bottleneck)
    elif args.provider == "hf":
        judge = HFJudge(model_repo=args.hf_model)
    elif args.provider == "local_cls":
        judge = LocalPhoBERTJudge(cls_model_repo=args.cls_model, span_model_repo=args.span_model)
    else:
        judge = LocalJudge(model_repo=args.hf_model, no_slang=args.no_slang, bottleneck=args.bottleneck)
        
    pipeline = CascadedPipeline(rated, judge, only_stage1=args.only_stage1, only_stage2=args.only_stage2)
    
    # Load Data
    dataset = InferenceDataset(DATA_PATH, rated.tokenizer, split='test', limit=args.limit)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    results = []
    
    start_time = time.time()
    
    # --- LOGGER SETUP ---
    class DualLogger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w", encoding='utf-8')
            
        def print(self, message):
            self.terminal.write(message + "\n")
            self.log.write(message + "\n")
            self.terminal.flush()
            self.log.flush() # Ensure write immediately

        def close(self):
            self.log.close()

    # Determine log path: có chú thích tên model, timestamp để không ghi đè
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    provider_name = args.provider
    if provider_name == "gemini":
        model_display = "Gemini-2.5-Flash-Lite"
        log_name = f"cascaded_results_vn_judge_{model_display}_{timestamp}.log"
    elif provider_name == "hf" or provider_name == "local":
        model_clean = os.path.basename(args.hf_model.rstrip("/\\"))
        model_display = model_clean
        log_name = f"cascaded_results_vn_judge_{provider_name}_{model_clean}_{timestamp}.log"
    elif provider_name == "local_cls":
        cls_clean = os.path.basename(args.cls_model.rstrip("/\\"))
        span_clean = os.path.basename(args.span_model.rstrip("/\\")) if args.span_model else "none"
        model_display = f"{cls_clean}+{span_clean}"
        log_name = f"cascaded_results_vn_judge_{provider_name}_{cls_clean}_{span_clean}_{timestamp}.log"
    else:
        model_display = provider_name
        log_name = f"cascaded_results_vn_judge_{provider_name}_{timestamp}.log"

    # Create Dynamic Result Dir based on Mode
    sub_dir = "cascaded"
    if args.only_stage1: sub_dir = "only_stage1"
    elif args.only_stage2: sub_dir = "only_stage2"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results", sub_dir)
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    provider_name = args.provider
    if provider_name == "gemini":
        model_display = "Gemini-2.5-Flash-Lite"
        log_name = f"cascaded_results_vn_judge_{model_display}_{timestamp}.log"
    elif provider_name == "hf" or provider_name == "local":
        model_clean = os.path.basename(args.hf_model.rstrip("/\\"))
        model_display = model_clean
        log_name = f"cascaded_results_vn_judge_{provider_name}_{model_clean}_{timestamp}.log"
    elif provider_name == "local_cls":
        cls_clean = os.path.basename(args.cls_model.rstrip("/\\"))
        span_clean = os.path.basename(args.span_model.rstrip("/\\")) if args.span_model else "none"
        model_display = f"{cls_clean}+{span_clean}"
        log_name = f"cascaded_results_vn_judge_{provider_name}_{cls_clean}_{span_clean}_{timestamp}.log"
    else:
        model_display = provider_name
        log_name = f"cascaded_results_vn_judge_{provider_name}_{timestamp}.log"

    log_filename = os.path.join(results_dir, log_name)
    logger = DualLogger(log_filename)
    logger.print(f"[Judge Model: {model_display}]")
    logger.print(f"[Log file: {log_name}]")

    # Also adjust args.output (JSON)
    if not os.path.isabs(args.output) and not os.path.dirname(args.output):
        base = os.path.splitext(args.output)[0]
        ext = os.path.splitext(args.output)[1] or ".json"
        
        if provider_name == "gemini":
            args.output = os.path.join(results_dir, f"{base}_gemini_{timestamp}{ext}")
        else:
            args.output = os.path.join(results_dir, f"{base}_{timestamp}{ext}")
    else:
        # If absolute path or has parent dir, try to honor it but put in sub_dir if simple filename
        pass

    print(f"Logging analysis to: {log_filename}")

    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        texts = batch['text']
        
        # Batch processing
        pipeline_outs = pipeline.process_batch(texts)
        
        for i, pipeline_out in enumerate(pipeline_outs):
            global_idx = batch_idx * args.batch_size + i
            raw_item = dataset.data[global_idx]
            
            results.append({
                "text": texts[i],
                "true_label": raw_item.get('label', 'N/A'),
                "pred_label": pipeline_out["label"],
                "pred_spans": pipeline_out["spans"],
                "pred_indices": pipeline_out["char_indices"],
                "gold_spans": raw_item.get('spans', []),
                "flow": pipeline_out["flow"],
                "rated_confidence": pipeline_out["original_rated"]["confidence"],
                "raw_item": raw_item
            })

    total_time = time.time() - start_time
    
    # Stats
    logger.print("\n" + "="*40)
    logger.print("CASCADED PIPELINE RESULTS")
    logger.print("="*40)
    logger.print(f"Total Processed: {pipeline.stats['total']}")
    logger.print(f"Time Taken: {total_time:.2f}s ({total_time/pipeline.stats['total']:.2f}s/sample)")
    logger.print("-" * 20)
    logger.print(f"RATeD Clean (Fast Path): {pipeline.stats['rated_clean']} ({pipeline.stats['rated_clean']/pipeline.stats['total']*100:.1f}%)")
    logger.print(f"Sent to Judge: {pipeline.stats['rated_verify']} ({pipeline.stats['rated_verify']/pipeline.stats['total']*100:.1f}%)")
    logger.print(f"  -> High Confidence Bypass (Trusted RATeD): {pipeline.stats['rated_high_conf_bypass']}")
    logger.print(f"  -> Judge Confirmed Toxic (Recovered FN): {pipeline.stats['confirmed_toxic']}")
    logger.print(f"  -> Judge Flipped to Clean (False Pos Saved): {pipeline.stats['judge_flipped_to_clean']}")
    logger.print(f"  -> Judge Confirmed Clean: {pipeline.stats['judge_confirmed_clean']}")
    
    # --- CALCULATE METRICS VS GROUND TRUTH ---
    from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
    
    
    # Metric Arrays
    y_true = []
    y_pred_e1 = []
    y_pred_e11_cls = []
    # Span Metric Aggregators
    all_syl_acc = []
    all_syl_f1 = []
    all_token_acc = []
    all_token_p = []
    all_token_r = []
    all_token_mf1 = []
    all_span_mf1 = []
    all_span_wf1 = []
    all_span_acc = []
    all_strict_f1 = []

    valid_entries = 0
    for r in results:
        # Normalize labels for ViHOS (safe/unsafe) or generic toxic labels
        # 1 = Toxic/Unsafe, 0 = Normal/Safe
        def normalize_lbl(l):
            l_str = str(l).strip().lower()
            if any(hit in l_str for hit in ["unsafe", "hate", "toxic", "offensive", "1"]):
                return 1
            return 0
            
        t_val = normalize_lbl(r['true_label'])
        p_val_e11 = normalize_lbl(r['pred_label'])
        
        # Determine E1 label (RATeD Only) - Baseline Comparison for curiosity
        p_val_e1 = 1 if r["rated_confidence"] > 0.5 else 0
        
        y_true.append(t_val)
        y_pred_e1.append(p_val_e1)
        y_pred_e11_cls.append(p_val_e11)
        
        # --- COLLECT METRICS ---
        # 1. Syllable
        s_acc, s_f1_bin, s_f1_macro = calculate_syllable_metrics(r['pred_spans'], r.get('gold_spans', []), r['text'])
        all_syl_acc.append(s_acc)
        all_syl_f1.append(s_f1_macro) # We use Macro F1 for consistency
        
        # 2. Token
        tok_m = calculate_token_metrics_full(r['pred_spans'], r.get('gold_spans', []), r['text'], rated.tokenizer)
        if tok_m:
            all_token_acc.append(tok_m['Token Accuracy'])
            all_token_p.append(tok_m['Token Precision'])
            all_token_r.append(tok_m['Token Recall'])
            all_token_mf1.append(tok_m['Token mF1'])
            
        # 3. Standardized Metric (ViHateT5 Style)
        # Use absolute character indices for precision
        text = r['text']
        
        # FIX: Prioritize unsafe_spans_indices which is standard in our jsonl
        raw_item = r.get('raw_item', {})
        if 'unsafe_spans_indices' in raw_item:
            gold_indices = get_char_indices_from_spans(raw_item['unsafe_spans_indices'])
        elif 'original_spans' in raw_item:
            try:
                gold_indices = set(ast.literal_eval(raw_item['original_spans']))
            except:
                gold_indices = set()
        else:
            gold_indices = set()
        
        # Mapping predicted indices
        pred_indices = set(r.get('pred_indices', []))
        
        c_acc, c_mf1, c_wf1, c_f1_bin = vihatet5_standardized_span_metrics(pred_indices, gold_indices, text)
        all_span_acc.append(c_acc)
        all_span_mf1.append(c_mf1)
        all_span_wf1.append(c_wf1)
        all_strict_f1.append(c_f1_bin)
        
        valid_entries += 1
        
    if valid_entries > 0:
        # --- AGGREGATE METRICS ---
        cls_metrics = {
            "Accuracy": accuracy_score(y_true, y_pred_e11_cls),
            "F1-Macro": f1_score(y_true, y_pred_e11_cls, average='macro', zero_division=0),
            "Precision": precision_score(y_true, y_pred_e11_cls, average='macro', zero_division=0),
            "Recall": recall_score(y_true, y_pred_e11_cls, average='macro', zero_division=0)
        }
        
        # Only report the 3 metrics from Table 3 (Hate Spans Detection)
        span_metrics = {
            "Acc": np.mean(all_span_acc) if all_span_acc else 0.0,
            "WF1": np.mean(all_span_wf1) if all_span_wf1 else 0.0,
            "MF1": np.mean(all_span_mf1) if all_span_mf1 else 0.0
        }
        
        cm = confusion_matrix(y_true, y_pred_e11_cls)
        
        # --- GENERATE STANDARDIZED REPORT ---
        report_name = f"RATeD-V E11 (Judge: {model_display})"
        report_str = generate_report_string(report_name, cls_metrics, span_metrics, cm)
        
        logger.print("\n" + report_str)
        
        # Also print time stats separately as they are pipeline specific
        logger.print(f"\nTime Stats: {total_time:.2f}s total | {total_time/pipeline.stats['total']:.2f}s/avg")

    
    # Save
    # Save
    # args.output is now processed to ideally be an absolute path in the correct results folder
    out_path = args.output
    # Fallback safety if not absolute (though handled above)
    if not os.path.dirname(out_path): 
         script_dir = os.path.dirname(os.path.abspath(__file__))
         out_path = os.path.join(script_dir, "results", out_path)
         
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.print(f"Results saved to {out_path}")
    logger.close()

if __name__ == "__main__":
    main()
