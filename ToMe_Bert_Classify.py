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
from sklearn.model_selection import train_test_split
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
            metric = k.mean(dim=1)                      # (B, T, d)
            merge_fn, _ = bipartite_soft_matching(metric, self.r)

            q = self._merge_heads(q, merge_fn)          # (B, H, T', d)
            k = self._merge_heads(k, merge_fn)
            v = self._merge_heads(v, merge_fn)

            # KEY FIX: merge residual to T' so sizes match for LayerNorm
            residual = merge_fn(residual)               # (B, T', C)
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
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # (B, 768)
        return self.classifier(cls)


# ─────────────────────────────────────────────
# 5. TRAINING & EVALUATION HELPERS
# ─────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    mode: str
    accuracy: float
    avg_inference_ms: float
    peak_memory_mb: float
    total_params: int


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
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
    return total_loss / len(loader)


def evaluate(model, loader, device) -> Tuple[float, float]:
    model.eval()
    all_preds, all_labels = [], []
    latencies = []

    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"]

            t0 = time.perf_counter()
            logits = model(ids, mask)
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

            preds = logits.argmax(dim=-1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(lbls.tolist())

    acc = accuracy_score(all_labels, all_preds)
    avg_ms = np.mean(latencies)
    return acc, avg_ms


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
    df: pd.DataFrame,
    num_epochs: int = 3,
    batch_size: int = 16,
    max_length: int = 256,
    tome_r: int = 8,
    learning_rate: float = 2e-5,
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
    df = load_and_preprocess(df)
    df = df.dropna(subset=["Label"])

    le = LabelEncoder()
    labels = le.fit_transform(df["Label"].astype(str))
    num_labels = len(le.classes_)
    print(f"Classes: {num_labels}  |  Samples: {len(df)}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(), labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_ds = PaperDataset(X_train, y_train, tokenizer, max_length)
    test_ds  = PaperDataset(X_test,  y_test,  tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    results = []
    for use_tome in [False, True]:
        label = "ToMe ON " if use_tome else "ToMe OFF"
        print(f"\n── {label} ──────────────────────────────────────────")

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        model = BertClassifier(num_labels, use_tome=use_tome, tome_r=tome_r).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            print(f"  Epoch {epoch+1}/{num_epochs}  loss={loss:.4f}")

        acc, avg_ms = evaluate(model, test_loader, device)
        mem = peak_memory_mb(device)
        params = count_params(model)

        result = BenchmarkResult(
            mode=label,
            accuracy=acc,
            avg_inference_ms=avg_ms,
            peak_memory_mb=mem,
            total_params=params,
        )
        results.append(result)
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Avg batch inference: {avg_ms:.2f} ms")
        print(f"  Peak GPU memory: {mem:.1f} MB")

    return results[0], results[1]


def print_comparison(baseline: BenchmarkResult, tome: BenchmarkResult):
    speedup  = baseline.avg_inference_ms / (tome.avg_inference_ms + 1e-9)
    acc_diff = (tome.accuracy - baseline.accuracy) * 100

    print("\n" + "=" * 55)
    print("  COMPARISON SUMMARY")
    print("=" * 55)
    print(f"{'Metric':<28} {'Baseline':>10} {'ToMe':>10}")
    print("-" * 55)
    print(f"{'Accuracy':<28} {baseline.accuracy:>10.4f} {tome.accuracy:>10.4f}")
    print(f"{'Avg Inference (ms)':<28} {baseline.avg_inference_ms:>10.2f} {tome.avg_inference_ms:>10.2f}")
    print(f"{'Peak GPU Memory (MB)':<28} {baseline.peak_memory_mb:>10.1f} {tome.peak_memory_mb:>10.1f}")
    print("-" * 55)
    print(f"  Speed-up factor   : {speedup:.2f}×")
    print(f"  Accuracy Δ        : {acc_diff:+.2f}%")
    print("=" * 55)


# ─────────────────────────────────────────────
# 7. ENTRY POINT – swap in your real DataFrame
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    train = pd.read_csv("D:\\File\\train_dataset\\train_set.csv")
    train_filter = train[(train["Label"] == 1) | (train["Label"] == 0)] # 636 + 729 + 516
    #print(train_filter.shape)
    samples = train_filter.sample(1000)
    df = pd.DataFrame(samples)

    baseline, tome = run_benchmark(
        df,
        num_epochs=3,     # ↑ for real training
        batch_size=8,
        max_length=128,   # ↑ to 256/512 for full abstracts
        tome_r=8,         # tokens merged per layer; try 4, 8, 16
    )

    print_comparison(baseline, tome)