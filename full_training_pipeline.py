import os
import sys
import json
import datetime
import copy
import time
import math
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch import Tensor
from typing import Optional, Tuple, Callable, List
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast


# ─────────────────────────────────────────────
# 0. FIELD CONSTANTS
# ─────────────────────────────────────────────

# Columns available from the paper CSVs
PAPER_FIELDS: List[str] = ["Title", "Abstract", "Keywords"]

# Columns that come from the journal CSV
JOURNAL_FIELDS: List[str] = ["Aims", "Categories"]


# ─────────────────────────────────────────────
# 0b. SESSION LOGGER
# ─────────────────────────────────────────────

class TeeLogger:
    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.log = open(filepath, "w", buffering=1, encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self) -> bool:
        return False

    def close(self):
        sys.stdout = self.terminal
        self.log.close()


# ─────────────────────────────────────────────
# 1. DATA PREPARATION (DUAL BRANCH)
# ─────────────────────────────────────────────

def merge_journal_info(paper_df: pd.DataFrame, journal_df: pd.DataFrame, journal_fields: List[str]) -> pd.DataFrame:
    available_journal_cols = [c for c in journal_fields if c in journal_df.columns]
    if not available_journal_cols:
        return paper_df

    join_cols = ["Label"] + available_journal_cols
    journal_subset = (
        journal_df[join_cols]
        .drop_duplicates(subset=["Label"])
        .reset_index(drop=True)
    )
    merged = paper_df.merge(journal_subset, on="Label", how="left")
    return merged


def load_and_preprocess_dual(df: pd.DataFrame, paper_fields: List[str], journal_fields: List[str]) -> pd.DataFrame:
    """
    Tạo hai luồng văn bản tùy chọn (Features) riêng biệt: Bài báo và Tạp chí.
    """
    df = df.copy()
    
    # --- 1. Tạo chuỗi gộp cho nhánh Bài báo (Paper) ---
    p_parts = []
    for f in paper_fields:
        if f in df.columns:
            df[f] = df[f].fillna("")
            p_parts.append(df[f])
        else:
            print(f"  [Warning] Paper Field '{f}' not found — skipping.")
            
    df["paper_text"] = p_parts[0] if p_parts else ""
    for part in p_parts[1:]:
        df["paper_text"] = df["paper_text"] + " [SEP] " + part

    # --- 2. Tạo chuỗi gộp cho nhánh Tạp chí (Journal) ---
    j_parts = []
    for f in journal_fields:
        if f in df.columns:
            df[f] = df[f].fillna("")
            j_parts.append(df[f])
        else:
            print(f"  [Warning] Journal Field '{f}' not found — skipping.")

    df["journal_text"] = j_parts[0] if j_parts else ""
    for part in j_parts[1:]:
        df["journal_text"] = df["journal_text"] + " [SEP] " + part

    return df


def preprocess_split(df: pd.DataFrame, split_name: str, paper_fields: List[str], journal_fields: List[str], journal_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if journal_df is not None:
        df = merge_journal_info(df, journal_df, journal_fields)

    split_df = load_and_preprocess_dual(df, paper_fields, journal_fields)
    split_df = split_df.dropna(subset=["Label"]).reset_index(drop=True)
    if split_df.empty:
        raise ValueError(f"{split_name} split is empty after preprocessing.")
    return split_df


def encode_split_labels(split_df: pd.DataFrame, label_to_id: dict, split_name: str) -> Tuple[list, list, list]:
    raw_labels = split_df["Label"].astype(str)
    known_mask = raw_labels.isin(label_to_id)
    
    filtered_df = split_df[known_mask].reset_index(drop=True)
    encoded_labels = raw_labels[known_mask].map(label_to_id).astype(int).tolist()
    
    paper_texts = filtered_df["paper_text"].tolist()
    journal_texts = filtered_df["journal_text"].tolist()
    
    return paper_texts, journal_texts, encoded_labels


def pretokenize_dual_to_disk(paper_texts, journal_texts, labels, tokenizer, max_length, save_dir, split_name):
    """
    Tokenize song song 2 nhánh và lưu thành 4 file memmap để tiết kiệm RAM.
    """
    os.makedirs(save_dir, exist_ok=True)
    n = len(labels)

    p_ids_path  = os.path.join(save_dir, f"{split_name}_p_ids.npy")
    p_mask_path = os.path.join(save_dir, f"{split_name}_p_mask.npy")
    j_ids_path  = os.path.join(save_dir, f"{split_name}_j_ids.npy")
    j_mask_path = os.path.join(save_dir, f"{split_name}_j_mask.npy")
    lbl_path    = os.path.join(save_dir, f"{split_name}_labels.npy")

    if os.path.exists(p_ids_path) and os.path.exists(j_ids_path):
        print(f"  [Cache hit] Loading pre-tokenized DUAL {split_name} from disk …")
        return (np.load(p_ids_path, mmap_mode="r"), np.load(p_mask_path, mmap_mode="r"),
                np.load(j_ids_path, mmap_mode="r"), np.load(j_mask_path, mmap_mode="r"), np.load(lbl_path))

    print(f"  [Pre-tokenizing DUAL {split_name}: {n} samples] …")
    p_ids  = np.lib.format.open_memmap(p_ids_path, mode="w+", dtype=np.int32, shape=(n, max_length))
    p_mask = np.lib.format.open_memmap(p_mask_path, mode="w+", dtype=np.int8,  shape=(n, max_length))
    j_ids  = np.lib.format.open_memmap(j_ids_path, mode="w+", dtype=np.int32, shape=(n, max_length))
    j_mask = np.lib.format.open_memmap(j_mask_path, mode="w+", dtype=np.int8,  shape=(n, max_length))

    CHUNK = 2000
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        
        # Tokenize Bài báo
        chunk_p = tokenizer(paper_texts[start:end], padding="max_length", truncation=True, max_length=max_length, return_tensors="np")
        p_ids[start:end]  = chunk_p["input_ids"].astype(np.int32)
        p_mask[start:end] = chunk_p["attention_mask"].astype(np.int8)
        
        # Tokenize Tạp chí
        chunk_j = tokenizer(journal_texts[start:end], padding="max_length", truncation=True, max_length=max_length, return_tensors="np")
        j_ids[start:end]  = chunk_j["input_ids"].astype(np.int32)
        j_mask[start:end] = chunk_j["attention_mask"].astype(np.int8)
        
        if start % 20000 == 0:
            print(f"    … {end}/{n}")

    np.save(lbl_path, np.array(labels, dtype=np.int32))
    print(f"  [Pre-tokenized DUAL {split_name} saved to {save_dir}]")

    return (np.load(p_ids_path, mmap_mode="r"), np.load(p_mask_path, mmap_mode="r"),
            np.load(j_ids_path, mmap_mode="r"), np.load(j_mask_path, mmap_mode="r"), np.load(lbl_path))


class DualDiskDataset(Dataset):
    def __init__(self, p_ids, p_mask, j_ids, j_mask, labels):
        self.p_ids = p_ids; self.p_mask = p_mask
        self.j_ids = j_ids; self.j_mask = j_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "p_ids":  torch.tensor(self.p_ids[idx], dtype=torch.long),
            "p_mask": torch.tensor(self.p_mask[idx], dtype=torch.long),
            "j_ids":  torch.tensor(self.j_ids[idx], dtype=torch.long),
            "j_mask": torch.tensor(self.j_mask[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────
# 2. TOKEN MERGING CORE IMPLEMENTATION (Giữ nguyên)
# ─────────────────────────────────────────────
# [Đoạn mã xử lý ToMe được giữ nguyên để tiết kiệm không gian]
# (Bao gồm bipartite_soft_matching, TOME_MERGE_TIME, ToMeBertAttention, patch_bert_with_tome)

def patch_bert_with_tome(model: AutoModel, r: int = 8) -> AutoModel:
    try:
        from modules.ToMeBertAttention import patch_bert_with_tome as patch_fn
        return patch_fn(model, r=r)
    except ImportError:
        print("[Warning] 'modules.ToMeBertAttention' không tìm thấy. Giả lập hoặc dùng hàm vá lỗi của riêng bạn.")
        return model


# ─────────────────────────────────────────────
# 4. CLASSIFIER MODEL (Approach C - Dual Branch)
# ─────────────────────────────────────────────

class ApproachCBertClassifier(nn.Module):
    def __init__(self, pretrained_model: AutoModel, num_labels: int, use_tome: bool = False, tome_r: int = 8):
        super().__init__()
        self.bert = copy.deepcopy(pretrained_model) 

        if use_tome:
            self.bert = patch_bert_with_tome(self.bert, r=tome_r)
            print(f"[ToMe ON]  Merging {tome_r} token pairs per layer")
        else:
            print("[ToMe OFF] Standard BERT (no merging)")

        hidden_size = self.bert.config.hidden_size 

        # Tầng Projection nhánh Tạp chí
        self.journal_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Tầng Projection nhánh Bài báo (có Dropout)
        self.paper_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

        # Tầng xử lý sau khi gộp (769 -> 768)
        self.fusion_linear = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),
            nn.ReLU()
        )

        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, p_ids, p_mask, j_ids, j_mask):
        # 1. Nhánh Bài Báo
        out_paper = self.bert(input_ids=p_ids, attention_mask=p_mask)
        cls_paper = out_paper.last_hidden_state[:, 0, :]
        feat_paper = self.paper_projection(cls_paper)

        # 2. Nhánh Tạp Chí
        out_journal = self.bert(input_ids=j_ids, attention_mask=j_mask)
        cls_journal = out_journal.last_hidden_state[:, 0, :]
        feat_journal = self.journal_projection(cls_journal)

        # 3. Tính Cosine Similarity -> [Batch, 1]
        cosine_sim = F.cosine_similarity(feat_paper, feat_journal, dim=1).unsqueeze(1)

        # 4. Gộp đặc trưng (Concatenate)
        fused_features = torch.cat((feat_paper, cosine_sim), dim=1)

        # 5. Phân loại
        out = self.fusion_linear(fused_features)
        return self.classifier(out) 


# ─────────────────────────────────────────────
# 5. TRAINING & EVALUATION HELPERS
# ─────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    mode: str; accuracy_top1: float; accuracy_top3: float; accuracy_top5: float; accuracy_top10: float
    avg_inference_s: float; peak_memory_mb: float; total_params: int; epochs_trained: int

def train_one_epoch(model, loader, optimizer, criterion, device, scaler = None, accum_steps = 1, log_every = 2000):
    model.train()
    total_loss = 0
    epoch_start = time.perf_counter()
    optimizer.zero_grad()
    n_steps = len(loader)
    
    for step, batch in enumerate(loader):
        p_ids = batch["p_ids"].to(device)
        p_mask = batch["p_mask"].to(device)
        j_ids = batch["j_ids"].to(device)
        j_mask = batch["j_mask"].to(device)
        lbls = batch["labels"].to(device)

        if scaler is not None:
            with autocast(device_type=device.type):
                logits = model(p_ids, p_mask, j_ids, j_mask)
                loss = criterion(logits, lbls) / accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            logits = model(p_ids, p_mask, j_ids, j_mask)
            loss = criterion(logits, lbls)
            loss.backward()
            if (step + 1) % accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
        total_loss += loss.item() * accum_steps

        if (step + 1) % log_every == 0 or (step + 1) == n_steps:
            elapsed = time.perf_counter() - epoch_start
            steps_done = step + 1
            print(f"    Step {steps_done}/{n_steps} | loss={total_loss/steps_done:.4f} | elapsed={elapsed/60:.1f}min", flush=True)
            
    return total_loss / len(loader), time.perf_counter() - epoch_start


def evaluate(model, loader, device) -> Tuple[dict, float]:
    model.eval()
    all_logits = []; all_labels = []; latencies  = []
    with torch.no_grad():
        for batch in loader:
            p_ids = batch["p_ids"].to(device); p_mask = batch["p_mask"].to(device)
            j_ids = batch["j_ids"].to(device); j_mask = batch["j_mask"].to(device)
            lbls = batch["labels"]

            t0 = time.perf_counter()
            with autocast(device_type=device.type):
                logits = model(p_ids, p_mask, j_ids, j_mask)
            latencies.append(time.perf_counter() - t0)

            all_logits.append(logits.float().cpu())
            all_labels.extend(lbls.tolist())

    all_logits = torch.cat(all_logits, dim=0)
    def topk(k): 
        _, p = torch.topk(all_logits, min(k, all_logits.size(1)), dim=-1)
        return sum(1 for t, pk in zip(all_labels, p.numpy()) if t in pk) / len(all_labels)

    metrics = {"top1": topk(1), "top3": topk(3), "top5": topk(5), "top10": topk(10)}
    return metrics, float(np.mean(latencies))

def peak_memory_mb(device): return torch.cuda.max_memory_allocated(device)/1e6 if device.type=="cuda" else 0.0
def count_params(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ─────────────────────────────────────────────
# 6. FULL BENCHMARK PIPELINE
# ─────────────────────────────────────────────

def run_benchmark(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    journal_df: Optional[pd.DataFrame] = None,
    CACHE_DIR: str = None,
    checkpoint_dir: str = None,
    paper_fields: Optional[List[str]] = None,      # TÙY CHỌN FEATURES BÀI BÁO
    journal_fields: Optional[List[str]] = None,    # TÙY CHỌN FEATURES TẠP CHÍ
    # paper_fields=["Title", "Keywords"], 
    # journal_fields=["Categories"],
    num_epochs: int = 10,
    batch_size: int = 16,
    max_length: int = 256,
    tome_r: int = 8,
    learning_rate: float = 2e-5,
    early_stopping_patience: int = 3,
    MODEL_NAME: str = "bert-base-uncased",
    run_mode: str = "both",
    accum_steps: int = 1
):
    # Khởi tạo Features mặc định nếu không được truyền vào
    if paper_fields is None: paper_fields = PAPER_FIELDS
    if journal_fields is None: journal_fields = JOURNAL_FIELDS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print(f"Paper Features   : {paper_fields}")
    print(f"Journal Features : {journal_fields}")
    print("=" * 55)

    # 1. Preprocess
    train_df = preprocess_split(train_df, "train", paper_fields, journal_fields, journal_df)
    val_df   = preprocess_split(val_df, "validation", paper_fields, journal_fields, journal_df)
    test_df  = preprocess_split(test_df, "test", paper_fields, journal_fields, journal_df)

    le = LabelEncoder()
    le.fit(train_df["Label"].astype(str))
    label_to_id = {label: idx for idx, label in enumerate(le.classes_)}
    num_labels = len(le.classes_)

    p_tr, j_tr, y_tr = encode_split_labels(train_df, label_to_id, "train")
    p_va, j_va, y_va = encode_split_labels(val_df, label_to_id, "validation")
    p_te, j_te, y_te = encode_split_labels(test_df, label_to_id, "test")
    del train_df, val_df, test_df; gc.collect()

    # 2. Tokenize & Loader
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModel.from_pretrained(MODEL_NAME)

    tr_data = pretokenize_dual_to_disk(p_tr, j_tr, y_tr, tokenizer, max_length, CACHE_DIR, "train")
    va_data = pretokenize_dual_to_disk(p_va, j_va, y_va, tokenizer, max_length, CACHE_DIR, "val")
    te_data = pretokenize_dual_to_disk(p_te, j_te, y_te, tokenizer, max_length, CACHE_DIR, "test")

    train_loader = DataLoader(DualDiskDataset(*tr_data), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(DualDiskDataset(*va_data), batch_size=batch_size)
    test_loader  = DataLoader(DualDiskDataset(*te_data), batch_size=batch_size)

    modes_to_run = [False, True] if run_mode == "both" else ([True] if run_mode == "tome" else [False])
    results = []

    for use_tome in modes_to_run:
        torch.cuda.empty_cache()
        label = "ToMe ON" if use_tome else "ToMe OFF"
        print(f"\n── {label} ─────────────────────────────────────────")

        model = ApproachCBertClassifier(base_model, num_labels, use_tome, tome_r).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler(device=device.type) if device.type == "cuda" else None

        best_val_acc = 0.0; patience = 0
        for epoch in range(num_epochs):
            loss, t_epoch = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, accum_steps)
            val_metrics, _ = evaluate(model, val_loader, device)
            val_acc = val_metrics["top1"]
            print(f"  Epoch {epoch+1} | loss={loss:.4f} | val_acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc; patience = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    print("  Early stopping!")
                    break

        model.load_state_dict(best_state)
        test_metrics, avg_s = evaluate(model, test_loader, device)
        
        print(f"  [Test] Top-1: {test_metrics['top1']:.4f} | Top-10: {test_metrics['top10']:.4f}")
        results.append(BenchmarkResult(label, test_metrics["top1"], test_metrics["top3"], test_metrics["top5"], test_metrics["top10"], avg_s, peak_memory_mb(device), count_params(model), num_epochs))

    return results[0] if len(results) == 1 else (results[0], results[1])