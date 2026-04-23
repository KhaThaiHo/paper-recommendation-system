"""
Token Merging (ToMe) for Academic Paper Journal Classification
==============================================================

PIPELINE OVERVIEW:
  1. Preprocess: Title + Abstract + Keywords → combined text
  2. Tokenize: text → token embeddings (via BERT)
  3. [ToMe ON]  Merge similar tokens at each transformer layer → fewer tokens
  4. [ToMe OFF] Standard attention over all tokens
  5. CLS token → Linear classifier → journal label prediction
  6. Compare: speed, memory, accuracy between both modes

WHAT TOKEN MERGING DOES:
  - At each BERT layer, finds pairs of tokens with highest cosine similarity
  - Merges (averages) those token pairs → reduces sequence length progressively
  - Fewer tokens = fewer attention computations = faster inference
  - Accuracy trade-off is usually small (< 1-2%)
"""

import time
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch import Tensor
from typing import Optional, Tuple, Callable
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────
# 1. DATA PREPARATION
# ─────────────────────────────────────────────

def load_and_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine Title + Abstract + Keywords into a single text field.
    Fill NaN keywords with empty string.
    """
    df = df.copy()
    df["Keywords"] = df["Keywords"].fillna("")
    df["text"] = (
        df["Title"].fillna("") + " [SEP] " +
        df["Abstract"].fillna("") + " [SEP] " +
        df["Keywords"]
    )
    return df


def preprocess_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Preprocess one split and validate required labels are present."""
    split_df = load_and_preprocess(df)
    split_df = split_df.dropna(subset=["Label"]).reset_index(drop=True)
    if split_df.empty:
        raise ValueError(f"{split_name} split is empty after preprocessing.")
    return split_df


def encode_split_labels(
    split_df: pd.DataFrame,
    label_to_id: dict,
    split_name: str,
) -> Tuple[list, list]:
    """Map labels using training-label vocabulary and drop unseen labels."""
    raw_labels = split_df["Label"].astype(str)
    known_mask = raw_labels.isin(label_to_id)
    dropped_count = int((~known_mask).sum())

    if dropped_count > 0:
        print(f"[Warning] Dropping {dropped_count} samples in {split_name} split with unseen labels.")

    filtered_df = split_df[known_mask].reset_index(drop=True)
    if filtered_df.empty:
        raise ValueError(f"{split_name} split has no samples with labels seen in training split.")

    encoded_labels = raw_labels[known_mask].map(label_to_id).astype(int).tolist()
    texts = filtered_df["text"].tolist()
    return texts, encoded_labels


class PaperDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ─────────────────────────────────────────────
# 2. TOKEN MERGING CORE IMPLEMENTATION
# ─────────────────────────────────────────────

def bipartite_soft_matching(
    metric: Tensor,      # (B, T, C) – token features used for similarity
    r: int,              # number of token pairs to merge per layer
) -> Tuple[Callable, Callable]:
    """
    Bipartite soft matching from the ToMe paper (Bolya et al., 2022).

    Splits tokens into two sets (A and B), finds the most similar
    cross-set pairs, and returns merge / unmerge functions.

    Returns:
        merge   – callable that reduces token count by r
        unmerge – callable that restores original positions (for viz)
    """
    B, T, _ = metric.shape
    # Protect against r being too large
    r = min(r, T // 2)

    with torch.no_grad():
        metric = metric / (metric.norm(dim=-1, keepdim=True) + 1e-6)  # L2 norm

        # Split into two halves: A (even indices), B (odd indices)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]  # (B, T/2, C)

        # Cosine similarity between every A–B pair
        scores = a @ b.transpose(-1, -2)  # (B, T/2, T/2)

        # For each token in A, find its best match in B
        node_max, node_idx = scores.max(dim=-1)   # (B, T/2)

        # Sort A tokens by their best-match score (descending)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., :r]  # (B, r)

        # The B tokens they match with
        unm_idx = node_max.argsort(dim=-1, descending=True)[..., r:]   # unmerged A
        src_idx = edge_idx                                               # merged A
        dst_idx = node_idx.gather(dim=-1, index=edge_idx)               # matched B

    def merge(x: Tensor, mode="mean") -> Tensor:
        """Merge r token pairs; returns tensor with T-r tokens."""
        src, dst = x[..., ::2, :], x[..., 1::2, :].clone()  # split A/B

        # Gather matched sources and accumulate into dst
        n, t1, c = src.shape
        matched_src = src.gather(
            dim=-2,
            index=src_idx.unsqueeze(-1).expand(n, r, c)
        )
        if mode == "mean":
            dst.scatter_reduce_(
                -2,
                dst_idx.unsqueeze(-1).expand(n, r, c),
                matched_src,
                reduce="mean",
                include_self=True,
            )
        # Unmerged A tokens
        unmerged = src.gather(
            dim=-2,
            index=unm_idx.unsqueeze(-1).expand(n, t1 - r, c)
        )
        # Concatenate unmerged A with updated B
        return torch.cat([unmerged, dst], dim=1)

    def unmerge(x: Tensor) -> Tensor:
        """Approximately restore original token positions (for inspection)."""
        # Simple pass-through; full reconstruction requires storing indices
        return x

    return merge, unmerge


# ─────────────────────────────────────────────
# 3. ToMe-PATCHED BERT SELF-ATTENTION
# ─────────────────────────────────────────────

# Variable to track total ToMe merge time across all calls for reporting after each epoch
TOME_MERGE_TIME = {"total_ms": 0.0, "call_count": 0}

def reset_tome_timer():
    """Reset the ToMe merge timing counters."""
    global TOME_MERGE_TIME
    TOME_MERGE_TIME = {"total_ms": 0.0, "call_count": 0}

def get_tome_timer_stats():
    """Get ToMe merge timing statistics."""
    global TOME_MERGE_TIME
    return TOME_MERGE_TIME.copy()

class ToMeBertAttention(nn.Module):
    """
    Replaces the ENTIRE BertAttention block (self-attn + output projection +
    residual + LayerNorm) so that Token Merging can also merge the residual
    tensor before the add — fixing the sequence-length mismatch:

        BertSelfOutput does:  LayerNorm( attn_out[T'] + residual[T] )
                                          T' != T after merging → CRASH

    By owning the full block we merge the residual to T' before adding.
    """

    def __init__(self, original_bert_attention, r: int = 8):
        super().__init__()
        self.r = r

        orig_self = original_bert_attention.self
        self.num_attention_heads = orig_self.num_attention_heads
        self.attention_head_size = orig_self.attention_head_size
        self.all_head_size       = orig_self.all_head_size
        self.query   = orig_self.query
        self.key     = orig_self.key
        self.value   = orig_self.value
        self.dropout = orig_self.dropout

        # BertSelfOutput weights
        self.out_dense     = original_bert_attention.output.dense
        self.out_dropout   = original_bert_attention.output.dropout
        self.out_LayerNorm = original_bert_attention.output.LayerNorm

    def _transpose(self, x: Tensor) -> Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    @staticmethod
    def _merge_heads(x: Tensor, merge_fn: Callable) -> Tensor:
        """(B, H, T, d) -> merge along T -> (B, H, T', d)"""
        B, H, T, d = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, H * d)
        x = merge_fn(x)
        T2 = x.size(1)
        return x.reshape(B, T2, H, d).permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        past_key_values=None,
        output_attentions: bool = False,
        **kwargs,
    ):
        residual = hidden_states                        # save for skip-connection

        q = self._transpose(self.query(hidden_states))  # (B, H, T, d)
        k = self._transpose(self.key(hidden_states))
        v = self._transpose(self.value(hidden_states))

        # ── Token Merging ────────────────────────────────────────────────
        T = hidden_states.size(1)
        if self.r > 0 and T > 2:
            global TOME_MERGE_TIME
            
            t_merge_start = time.perf_counter()
            
            metric = k.mean(dim=1)                      # (B, T, d)
            merge_fn, _ = bipartite_soft_matching(metric, self.r)

            q = self._merge_heads(q, merge_fn)          # (B, H, T', d)
            k = self._merge_heads(k, merge_fn)
            v = self._merge_heads(v, merge_fn)

            # KEY FIX: merge residual to T' so sizes match for LayerNorm
            residual = merge_fn(residual)               # (B, T', C)
            
            t_merge_end = time.perf_counter()
            merge_time_ms = (t_merge_end - t_merge_start) * 1000
            TOME_MERGE_TIME["total_ms"] += merge_time_ms
            TOME_MERGE_TIME["call_count"] += 1
        # ────────────────────────────────────────────────────────────────

        scale  = math.sqrt(self.attention_head_size)
        scores = torch.matmul(q, k.transpose(-1, -2)) / scale
        probs  = nn.functional.softmax(scores, dim=-1)
        probs  = self.dropout(probs)
        ctx    = torch.matmul(probs, v)                 # (B, H, T', d)

        ctx = ctx.permute(0, 2, 1, 3).contiguous()
        ctx = ctx.view(ctx.size(0), ctx.size(1), self.all_head_size)

        # BertSelfOutput: dense -> dropout -> LayerNorm(x + residual)
        ctx = self.out_dense(ctx)
        ctx = self.out_dropout(ctx)
        attention_output = self.out_LayerNorm(ctx + residual)  # both T'

        attn_weights = probs if output_attentions else None
        return (attention_output, attn_weights)                 # always 2-tuple


def patch_bert_with_tome(model: BertModel, r: int = 8) -> BertModel:
    """
    Replace every BertAttention block with ToMeBertAttention.
    Patching the full block (not just self-attn) is required so that
    the residual connection is also merged to match the reduced T'.
    """
    for layer in model.encoder.layer:
        layer.attention = ToMeBertAttention(layer.attention, r=r)
    return model


# ─────────────────────────────────────────────
# 4. CLASSIFIER MODEL
# ─────────────────────────────────────────────

class BertClassifier(nn.Module):
    def __init__(self, num_labels: int, use_tome: bool = False, tome_r: int = 8):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        if use_tome:
            self.bert = patch_bert_with_tome(self.bert, r=tome_r)
            print(f"[ToMe ON]  Merging {tome_r} token pairs per layer")
        else:
            print("[ToMe OFF] Standard BERT (no merging)")

        hidden_size = self.bert.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        # CLS token representation
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask) # Last hidden states shape: (B, T', 768)
        cls = out.last_hidden_state[:, 0, :]   # (B, 768)
        return self.classifier(cls)

# input_ids: [B, T] (Batch size, Sequence length)
# attention_mask: [B, T]
# BERT last_hidden_state: [B, T', 768]
# CLS: [B, 768]
# logits: [B, num_labels]
# ─────────────────────────────────────────────
# 5. TRAINING & EVALUATION HELPERS
# ─────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    mode: str
    accuracy_top1: float
    accuracy_top3: float
    accuracy_top5: float
    accuracy_top10: float
    avg_inference_ms: float
    peak_memory_mb: float
    total_params: int
    epochs_trained: int


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    epoch_start = time.perf_counter()
    
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)
        optimizer.zero_grad()
        logits = model(ids, mask)
        loss = criterion(logits, lbls)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    
    epoch_time = time.perf_counter() - epoch_start
    return total_loss / len(loader), epoch_time


def compute_topk_accuracy(all_logits: torch.Tensor, all_labels: list, k: int) -> float:
    """Compute top-k accuracy."""
    _, topk_preds = torch.topk(all_logits, min(k, all_logits.size(1)), dim=-1)
    topk_preds = topk_preds.cpu().numpy()
    correct = 0
    for true_label, pred_k in zip(all_labels, topk_preds):
        if true_label in pred_k:
            correct += 1
    return correct / len(all_labels)


def evaluate(model, loader, device) -> Tuple[dict, float]:
    model.eval()
    all_logits = []
    all_labels = []
    latencies = []

    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"]

            t0 = time.perf_counter()
            logits = model(ids, mask)
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

            all_logits.append(logits.cpu())
            all_labels.extend(lbls.tolist())

    all_logits = torch.cat(all_logits, dim=0)
    
    # Compute top-k accuracies
    acc_top1 = compute_topk_accuracy(all_logits, all_labels, 1)
    acc_top3 = compute_topk_accuracy(all_logits, all_labels, 3)
    acc_top5 = compute_topk_accuracy(all_logits, all_labels, 5)
    acc_top10 = compute_topk_accuracy(all_logits, all_labels, 10)
    
    avg_ms = np.mean(latencies)
    
    metrics = {
        'top1': acc_top1,
        'top3': acc_top3,
        'top5': acc_top5,
        'top10': acc_top10,
    }
    return metrics, avg_ms


def peak_memory_mb(device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e6
    return 0.0  # CPU memory tracking not built-in


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# 6. FULL BENCHMARK PIPELINE
# ─────────────────────────────────────────────

def run_benchmark(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_epochs: int = 10,
    batch_size: int = 16,
    max_length: int = 256,
    tome_r: int = 8,
    learning_rate: float = 2e-5,
    early_stopping_patience: int = 3,
) -> Tuple[BenchmarkResult, BenchmarkResult]:
    """
    Trains and evaluates two models:
      - Baseline BERT (no ToMe)
      - BERT + Token Merging

    Returns both BenchmarkResult objects for comparison.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n{'='*55}")

    # ── Preprocess ────────────────────────────────────────────
    t_preprocess = time.perf_counter()
    train_df = preprocess_split(train_df, "train")
    val_df = preprocess_split(val_df, "validation")
    test_df = preprocess_split(test_df, "test")

    le = LabelEncoder()
    le.fit(train_df["Label"].astype(str))
    label_to_id = {label: idx for idx, label in enumerate(le.classes_)}

    X_train, y_train = encode_split_labels(train_df, label_to_id, "train")
    X_val, y_val = encode_split_labels(val_df, label_to_id, "validation")
    X_test, y_test = encode_split_labels(test_df, label_to_id, "test")

    num_labels = len(le.classes_)

    print(f"Classes: {num_labels} | Train: {len(y_train)} "
          f"| Val: {len(y_val)} | Test: {len(y_test)}")
    
    t_tokenize = time.perf_counter()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print(f"  [Loaded tokenizer]: {time.perf_counter() - t_tokenize:.2f}s")

    # ── Create Datasets ────────────────────────────────────────
    t_dataset = time.perf_counter()
    train_ds = PaperDataset(X_train, y_train, tokenizer, max_length)
    val_ds   = PaperDataset(X_val,   y_val,   tokenizer, max_length)
    test_ds  = PaperDataset(X_test,  y_test,  tokenizer, max_length)
    print(f"  [Created datasets]: {time.perf_counter() - t_dataset:.2f}s")
    
    t_loader = time.perf_counter()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)
    print(f"  [Created dataloaders]: {time.perf_counter() - t_loader:.2f}s")
    print(f"  Total preprocessing & data prep: {time.perf_counter() - t_preprocess:.2f}s\n")

    results = []
    for use_tome in [False, True]:
        label = "ToMe ON " if use_tome else "ToMe OFF"
        print(f"\n── {label} ──────────────────────────────────────────")

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # ── Initialize Model ────────────────────────────────────
        t_init = time.perf_counter()
        model = BertClassifier(num_labels, use_tome=use_tome, tome_r=tome_r).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        t_init_ms = (time.perf_counter() - t_init) * 1000
        print(f"  [Model initialized]: {t_init_ms:.2f}ms")
        
        total_train_time = 0
        best_val_acc = 0
        patience_counter = 0
        best_epoch = 0
        epochs_trained = 0
        best_model_state = None

        # ── Training Loop ────────────────────────────────────────
        t_training_start = time.perf_counter()
        for epoch in range(num_epochs):
            # Reset ToMe timer
            reset_tome_timer()
            
            loss, epoch_time = train_one_epoch(model, train_loader, optimizer, criterion, device)
            total_train_time += epoch_time
            epochs_trained += 1
            
            # Validation
            t_eval = time.perf_counter()
            val_metrics, _ = evaluate(model, val_loader, device)
            eval_time_ms = (time.perf_counter() - t_eval) * 1000
            val_acc = val_metrics['top1']
            
            # Get ToMe stats if using ToMe
            tome_stats = get_tome_timer_stats()
            tome_info = f"  ToMe merge: {tome_stats['total_ms']:.2f}ms ({tome_stats['call_count']} calls)" if tome_stats['call_count'] > 0 else ""
            
            print(f"  Epoch {epoch+1}/{num_epochs}  loss={loss:.4f}  train={epoch_time:.2f}s  eval={eval_time_ms:.2f}ms  val_acc={val_acc:.4f}{tome_info}", end="")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_epoch = epoch + 1
                best_model_state = model.state_dict().copy()
                print(" ✓")
            else:
                patience_counter += 1
                print(f" (patience: {patience_counter}/{early_stopping_patience})")
                if patience_counter >= early_stopping_patience:
                    print(f"  Early stopping at epoch {epoch+1} (best epoch: {best_epoch})")
                    break

        t_training_end = time.perf_counter()
        training_duration = t_training_end - t_training_start
        print(f"  [Training completed]: {training_duration:.2f}s ({epochs_trained} epochs)")

        # Load best model state before testing
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # ── Test Evaluation ────────────────────────────────────
        t_test = time.perf_counter()
        test_metrics, avg_ms = evaluate(model, test_loader, device)
        test_duration = (time.perf_counter() - t_test) * 1000
        print(f"  [Testing completed]: {test_duration:.2f}ms")
        
        mem = peak_memory_mb(device)
        params = count_params(model)

        result = BenchmarkResult(
            mode=label,
            accuracy_top1=test_metrics['top1'],
            accuracy_top3=test_metrics['top3'],
            accuracy_top5=test_metrics['top5'],
            accuracy_top10=test_metrics['top10'],
            avg_inference_ms=avg_ms,
            peak_memory_mb=mem,
            total_params=params,
            epochs_trained=epochs_trained,
        )
        results.append(result)
        
        print(f"\n  Test Results:")
        print(f"    Top-1  Accuracy: {test_metrics['top1']:.4f}")
        print(f"    Top-3  Accuracy: {test_metrics['top3']:.4f}")
        print(f"    Top-5  Accuracy: {test_metrics['top5']:.4f}")
        print(f"    Top-10 Accuracy: {test_metrics['top10']:.4f}")
        print(f"    Avg batch inference: {avg_ms:.2f} ms")
        print(f"    Peak GPU memory: {mem:.1f} MB")
        print(f"    Total training time: {total_train_time:.2f}s")
        print(f"    Epochs trained: {epochs_trained}/{num_epochs}")

    return results[0], results[1]


def print_comparison(baseline: BenchmarkResult, tome: BenchmarkResult):
    speedup  = baseline.avg_inference_ms / (tome.avg_inference_ms + 1e-9)

    print("\n" + "=" * 75)
    print("  COMPARISON SUMMARY - TOP-K ACCURACY TABLE")
    print("=" * 75)
    print(f"{'Metric':<25} {'Baseline':>15} {'ToMe':>15} {'Δ':>15}")
    print("-" * 75)
    
    # Top-1
    delta_top1 = (tome.accuracy_top1 - baseline.accuracy_top1) * 100
    print(f"{'Top-1 Accuracy':<25} {baseline.accuracy_top1:>15.4f} {tome.accuracy_top1:>15.4f} {delta_top1:>14.2f}%")
    
    # Top-3
    delta_top3 = (tome.accuracy_top3 - baseline.accuracy_top3) * 100
    print(f"{'Top-3 Accuracy':<25} {baseline.accuracy_top3:>15.4f} {tome.accuracy_top3:>15.4f} {delta_top3:>14.2f}%")
    
    # Top-5
    delta_top5 = (tome.accuracy_top5 - baseline.accuracy_top5) * 100
    print(f"{'Top-5 Accuracy':<25} {baseline.accuracy_top5:>15.4f} {tome.accuracy_top5:>15.4f} {delta_top5:>14.2f}%")
    
    # Top-10
    delta_top10 = (tome.accuracy_top10 - baseline.accuracy_top10) * 100
    print(f"{'Top-10 Accuracy':<25} {baseline.accuracy_top10:>15.4f} {tome.accuracy_top10:>15.4f} {delta_top10:>14.2f}%")
    
    print("-" * 75)
    print(f"{'Avg Inference (ms)':<25} {baseline.avg_inference_ms:>15.2f} {tome.avg_inference_ms:>15.2f} {(tome.avg_inference_ms - baseline.avg_inference_ms):>14.2f}")
    print(f"{'Peak GPU Memory (MB)':<25} {baseline.peak_memory_mb:>15.1f} {tome.peak_memory_mb:>15.1f} {(tome.peak_memory_mb - baseline.peak_memory_mb):>14.1f}")
    print(f"{'Epochs Trained':<25} {baseline.epochs_trained:>15} {tome.epochs_trained:>15}")
    print("-" * 75)
    print(f"  Speed-up factor: {speedup:.2f}×")
    print("=" * 75)


# ─────────────────────────────────────────────
# 7. ENTRY POINT – swap in your real DataFrame
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import pandas as pd
    script_start = time.perf_counter()

    parser = argparse.ArgumentParser(description="Benchmark BERT with or without ToMe using fixed data splits.")
    parser.add_argument("--train_csv", default="train_dataset.csv", help="Path to training CSV file")
    parser.add_argument("--val_csv", default="val_dataset.csv", help="Path to validation CSV file")
    parser.add_argument("--test_csv", default="test_dataset.csv", help="Path to test CSV file")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--tome_r", type=int, default=8)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)

    baseline, tome = run_benchmark(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        tome_r=args.tome_r,
        early_stopping_patience=args.early_stopping_patience,
    )

    print_comparison(baseline, tome)
    
    total_time = time.perf_counter() - script_start
    print(f"\n{'='*55}")
    print(f"  TOTAL SCRIPT EXECUTION TIME: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"{'='*55}")