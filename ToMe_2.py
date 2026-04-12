"""
Token Merging (ToMe) for Academic Paper Journal Classification
==============================================================
All 10 fixes applied:
  Fix 1  - No data leakage: LabelEncoder fit only on y_train
  Fix 2  - unmerge() correctly reconstructs token positions
  Fix 3  - Closure tensors freed after use (.contiguous() + del)
  Fix 4  - Configurable ToMe metric: keys | queries | values | hidden
  Fix 5  - Best model checkpoint saved and reloaded before test eval
  Fix 6  - Stratified val/test split with safe fallback
  Fix 7  - WeightedRandomSampler for class imbalance
  Fix 8  - Fair benchmark: shared BERT init weights + fixed seed
  Fix 9  - Cosine LR schedule with linear warmup (step per batch)
  Fix 10 - CPU memory tracking via tracemalloc
"""

import os
import time
import math
import tracemalloc
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch import Tensor
from typing import Optional, Tuple, Callable
from dataclasses import dataclass
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    BertModel,
    get_cosine_schedule_with_warmup,
)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ─────────────────────────────────────────────
# 1. DATA PREPARATION
# ─────────────────────────────────────────────

def load_and_preprocess(df: pd.DataFrame) -> pd.DataFrame:
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
# FIX 6: Safe stratified split
# ─────────────────────────────────────────────

def safe_stratified_split(
    X, y,
    test_size=0.5,
    random_state=42,
    min_class_count=2,
):
    """
    Stratified split with fallback:
    classes with fewer than min_class_count samples in this set
    are dropped before splitting to avoid sklearn ValueError.
    """
    counts = Counter(y)
    can_stratify = all(c >= min_class_count for c in counts.values())

    if can_stratify:
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

    valid = {label for label, c in counts.items() if c >= min_class_count}
    dropped = sum(1 for label in y if label not in valid)
    if dropped:
        print(f"  [Warning] Dropped {dropped} samples "
              f"(classes with <{min_class_count} examples in temp set)")
    X_f = [x for x, label in zip(X, y) if label in valid]
    y_f = [label for label in y if label in valid]
    return train_test_split(
        X_f, y_f,
        test_size=test_size,
        random_state=random_state,
        stratify=y_f,
    )


# ─────────────────────────────────────────────
# FIX 7: Weighted sampler for class imbalance
# ─────────────────────────────────────────────

def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Inverse-frequency weighting so rare classes are sampled fairly."""
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.where(class_counts > 0, class_counts, 1)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


# ─────────────────────────────────────────────
# 2. TOKEN MERGING CORE  (Fix 2 + Fix 3)
# ─────────────────────────────────────────────

def bipartite_soft_matching(
    metric: Tensor,
    r: int,
) -> Tuple[Callable, Callable]:
    """
    FIX 2: unmerge() now correctly reconstructs token positions.
    FIX 3: intermediate tensors freed; closure only holds .contiguous() copies.
    """
    B, T, _ = metric.shape
    r = min(r, T // 2)

    with torch.no_grad():
        metric_n = metric / (metric.norm(dim=-1, keepdim=True) + 1e-6)
        a, b     = metric_n[..., ::2, :], metric_n[..., 1::2, :]
        scores   = a @ b.transpose(-1, -2)

        node_max, node_idx = scores.max(dim=-1)
        order    = node_max.argsort(dim=-1, descending=True)
        edge_idx = order[..., :r]
        unm_idx  = order[..., r:]
        dst_idx  = node_idx.gather(dim=-1, index=edge_idx)

        # FIX 3: contiguous copies so originals can be freed
        _src_idx = edge_idx.contiguous()
        _dst_idx = dst_idx.contiguous()
        _unm_idx = unm_idx.contiguous()

        del metric_n, a, b, scores, node_max, node_idx, order
        del edge_idx, dst_idx, unm_idx

    T_a = (T + 1) // 2   # tokens in set A

    def merge(x: Tensor, mode: str = "mean") -> Tensor:
        """Reduce token count by r. Returns (B, T-r, C)."""
        src, dst = x[..., ::2, :], x[..., 1::2, :].clone()
        n, t1, c = src.shape

        matched = src.gather(
            dim=-2,
            index=_src_idx.unsqueeze(-1).expand(n, r, c),
        )
        if mode == "mean":
            dst.scatter_reduce_(
                -2,
                _dst_idx.unsqueeze(-1).expand(n, r, c),
                matched,
                reduce="mean",
                include_self=True,
            )
        unmerged = src.gather(
            dim=-2,
            index=_unm_idx.unsqueeze(-1).expand(n, t1 - r, c),
        )
        return torch.cat([unmerged, dst], dim=1)

    def unmerge(x: Tensor) -> Tensor:
        """
        FIX 2: Approximate reconstruction (B, T', C) → (B, T, C).
        Layout of x after merge:
          [:, :T_a-r, :]  = unmerged A tokens
          [:, T_a-r:, :]  = B tokens (some accumulated src)
        """
        n, _, c = x.shape
        T_a_unm = T_a - r

        unm_tokens = x[:, :T_a_unm, :]
        dst_tokens = x[:, T_a_unm:, :]

        out = torch.zeros(n, T, c, device=x.device, dtype=x.dtype)

        # Restore B tokens to odd positions
        out[:, 1::2, :] = dst_tokens

        # Restore unmerged A tokens to their original even positions
        out.scatter_(
            dim=1,
            index=(_unm_idx * 2).unsqueeze(-1).expand(n, T_a_unm, c),
            src=unm_tokens,
        )

        # Copy merged values back to src positions (approximate)
        merged_vals = dst_tokens.gather(
            dim=1,
            index=_dst_idx.unsqueeze(-1).expand(n, r, c),
        )
        out.scatter_(
            dim=1,
            index=(_src_idx * 2).unsqueeze(-1).expand(n, r, c),
            src=merged_vals,
        )

        return out

    return merge, unmerge


# ─────────────────────────────────────────────
# 3. ToMe-PATCHED BERT SELF-ATTENTION  (Fix 4)
# ─────────────────────────────────────────────

class ToMeBertAttention(nn.Module):
    """
    Replaces the full BertAttention block so the residual connection
    can be merged to T' before the LayerNorm add — fixing the
    sequence-length mismatch that would otherwise crash.

    FIX 4: tome_metric is now configurable.
    """

    def __init__(
        self,
        original_bert_attention,
        r: int = 8,
        tome_metric: str = "keys",   # "keys" | "queries" | "values" | "hidden"
    ):
        super().__init__()
        self.r = r
        self.tome_metric = tome_metric

        orig_self = original_bert_attention.self
        self.num_attention_heads = orig_self.num_attention_heads
        self.attention_head_size = orig_self.attention_head_size
        self.all_head_size        = orig_self.all_head_size
        self.query   = orig_self.query
        self.key     = orig_self.key
        self.value   = orig_self.value
        self.dropout = orig_self.dropout

        self.out_dense     = original_bert_attention.output.dense
        self.out_dropout   = original_bert_attention.output.dropout
        self.out_LayerNorm = original_bert_attention.output.LayerNorm

    def _transpose(self, x: Tensor) -> Tensor:
        new_shape = x.size()[:-1] + (
            self.num_attention_heads, self.attention_head_size
        )
        return x.view(*new_shape).permute(0, 2, 1, 3)

    @staticmethod
    def _merge_heads(x: Tensor, merge_fn: Callable) -> Tensor:
        B, H, T, d = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, H * d)
        x = merge_fn(x)
        T2 = x.size(1)
        return x.reshape(B, T2, H, d).permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        **kwargs,
    ):
        residual = hidden_states

        q = self._transpose(self.query(hidden_states))
        k = self._transpose(self.key(hidden_states))
        v = self._transpose(self.value(hidden_states))

        T = hidden_states.size(1)
        if self.r > 0 and T > 2:
            # FIX 4: configurable metric
            if self.tome_metric == "keys":
                metric = k.mean(dim=1)
            elif self.tome_metric == "queries":
                metric = q.mean(dim=1)
            elif self.tome_metric == "values":
                metric = v.mean(dim=1)
            elif self.tome_metric == "hidden":
                metric = hidden_states
            else:
                raise ValueError(f"Unknown tome_metric: {self.tome_metric}")

            merge_fn, _ = bipartite_soft_matching(metric, self.r)
            q = self._merge_heads(q, merge_fn)
            k = self._merge_heads(k, merge_fn)
            v = self._merge_heads(v, merge_fn)
            residual = merge_fn(residual)

        scale  = math.sqrt(self.attention_head_size)
        scores = torch.matmul(q, k.transpose(-1, -2)) / scale
        probs  = nn.functional.softmax(scores, dim=-1)
        probs  = self.dropout(probs)
        ctx    = torch.matmul(probs, v)

        ctx = ctx.permute(0, 2, 1, 3).contiguous()
        ctx = ctx.view(ctx.size(0), ctx.size(1), self.all_head_size)

        ctx = self.out_dense(ctx)
        ctx = self.out_dropout(ctx)
        attention_output = self.out_LayerNorm(ctx + residual)

        return (attention_output, None)


def patch_bert_with_tome(
    model: BertModel,
    r: int = 8,
    tome_metric: str = "keys",   # FIX 4
) -> BertModel:
    for layer in model.encoder.layer:
        layer.attention = ToMeBertAttention(
            layer.attention, r=r, tome_metric=tome_metric
        )
    return model


# ─────────────────────────────────────────────
# 4. CLASSIFIER MODEL
# ─────────────────────────────────────────────

class BertClassifier(nn.Module):
    def __init__(
        self,
        num_labels: int,
        use_tome: bool = False,
        tome_r: int = 8,
        tome_metric: str = "keys",   # FIX 4
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        if use_tome:
            self.bert = patch_bert_with_tome(
                self.bert, r=tome_r, tome_metric=tome_metric
            )
            print(f"[ToMe ON]  r={tome_r}, metric={tome_metric}")
        else:
            print("[ToMe OFF] Standard BERT")

        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)


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
    best_epoch: int


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    t0 = time.perf_counter()

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
        scheduler.step()   # FIX 9: step per batch
        total_loss += loss.item()

    return total_loss / len(loader), time.perf_counter() - t0


def compute_topk_accuracy(logits: torch.Tensor, labels: list, k: int) -> float:
    _, topk = torch.topk(logits, min(k, logits.size(1)), dim=-1)
    topk = topk.cpu().numpy()
    return sum(1 for true, pred in zip(labels, topk) if true in pred) / len(labels)


def evaluate(model, loader, device) -> Tuple[dict, float]:
    model.eval()
    all_logits, all_labels, latencies = [], [], []

    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"]

            t0 = time.perf_counter()
            logits = model(ids, mask)
            latencies.append((time.perf_counter() - t0) * 1000)

            all_logits.append(logits.cpu())
            all_labels.extend(lbls.tolist())

    all_logits = torch.cat(all_logits, dim=0)
    return {
        "top1":  compute_topk_accuracy(all_logits, all_labels, 1),
        "top3":  compute_topk_accuracy(all_logits, all_labels, 3),
        "top5":  compute_topk_accuracy(all_logits, all_labels, 5),
        "top10": compute_topk_accuracy(all_logits, all_labels, 10),
    }, float(np.mean(latencies))


# FIX 10: CPU memory via tracemalloc
def measure_peak_memory_mb(device, snapshot=None) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e6
    if snapshot is not None:
        _, peak = snapshot
        return peak / 1e6
    return 0.0


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# FIX 8: Shared BERT init for fair comparison
# ─────────────────────────────────────────────

_pretrained_cache: dict = {}

def get_fresh_model(
    num_labels: int,
    use_tome: bool,
    tome_r: int,
    tome_metric: str,
    seed: int = 42,
) -> BertClassifier:
    """
    FIX 8: Both baseline and ToMe start from identical BERT weights.
    Classifier head is reset with a fixed seed each time.
    """
    global _pretrained_cache

    if "bert_state" not in _pretrained_cache:
        tmp = BertModel.from_pretrained("bert-base-uncased")
        _pretrained_cache["bert_state"] = {
            k: v.clone() for k, v in tmp.state_dict().items()
        }
        del tmp
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    model = BertClassifier(
        num_labels, use_tome=use_tome,
        tome_r=tome_r, tome_metric=tome_metric,
    )
    model.bert.load_state_dict(_pretrained_cache["bert_state"], strict=False)

    torch.manual_seed(seed)
    for m in model.classifier.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    return model


# ─────────────────────────────────────────────
# 6. FULL BENCHMARK PIPELINE
# ─────────────────────────────────────────────

def run_benchmark(
    df: pd.DataFrame,
    num_epochs: int = 10,
    batch_size: int = 16,
    max_length: int = 256,
    tome_r: int = 8,
    tome_metric: str = "keys",       # FIX 4
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,      # FIX 9
    warmup_ratio: float = 0.1,       # FIX 9
    early_stopping_patience: int = 3,
    save_dir: str = "./checkpoints",
    seed: int = 42,
) -> Tuple[BenchmarkResult, BenchmarkResult]:

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cpu = device.type == "cpu"
    print(f"\nDevice: {device}\n{'='*60}")

    # ── Preprocess ───────────────────────────────────────────────
    df = load_and_preprocess(df)
    df = df.dropna(subset=["Label"])

    # FIX 1: work with raw string labels, split BEFORE encoding
    raw_labels = df["Label"].astype(str).tolist()

    # Filter classes with fewer than 3 samples (using Counter, no encoder)
    counts = Counter(raw_labels)
    valid_set = {l for l, c in counts.items() if c >= 3}
    mask = [l in valid_set for l in raw_labels]
    df = df[mask].reset_index(drop=True)
    raw_labels = [l for l, m in zip(raw_labels, mask) if m]
    print(f"After filtering — classes: {len(valid_set)} | samples: {len(df)}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # FIX 1 + FIX 6: split on raw strings, stratify both times
    texts = df["text"].tolist()
    X_train, X_temp, y_train_raw, y_temp_raw = train_test_split(
        texts, raw_labels,
        test_size=0.4, random_state=seed, stratify=raw_labels,
    )
    # FIX 6: safe stratified split for val/test
    X_val, X_test, y_val_raw, y_test_raw = safe_stratified_split(
        X_temp, y_temp_raw, test_size=0.5, random_state=seed,
    )

    # FIX 1: fit encoder ONLY on train labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val   = le.transform(y_val_raw)
    y_test  = le.transform(y_test_raw)
    num_labels = len(le.classes_)
    print(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}"
          f" | Classes: {num_labels}")

    train_ds = PaperDataset(X_train, y_train, tokenizer, max_length)
    val_ds   = PaperDataset(X_val,   y_val,   tokenizer, max_length)
    test_ds  = PaperDataset(X_test,  y_test,  tokenizer, max_length)

    # FIX 7: weighted sampler replaces shuffle=True
    train_sampler = make_weighted_sampler(y_train)
    train_loader  = DataLoader(train_ds, batch_size=batch_size,
                               sampler=train_sampler)
    val_loader    = DataLoader(val_ds,   batch_size=batch_size)
    test_loader   = DataLoader(test_ds,  batch_size=batch_size)

    results = []

    for use_tome in [False, True]:
        run_label = "tome" if use_tome else "baseline"
        print(f"\n── {run_label.upper()} ──────────────────────────────────────")

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # FIX 10: start CPU memory tracking
        cpu_snapshot = None
        if is_cpu:
            tracemalloc.start()

        # FIX 8: both models start from identical weights
        torch.manual_seed(seed)
        model = get_fresh_model(
            num_labels, use_tome, tome_r, tome_metric, seed
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,   # FIX 9
        )
        criterion = nn.CrossEntropyLoss()

        # FIX 9: cosine schedule with linear warmup
        total_steps  = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        best_val_acc   = 0.0
        patience_count = 0
        best_epoch_num = 0
        epochs_trained = 0
        total_train_t  = 0.0
        best_ckpt_path = os.path.join(save_dir, f"best_{run_label}.pt")

        for epoch in range(num_epochs):
            loss, epoch_t = train_one_epoch(
                model, train_loader, optimizer, scheduler, criterion, device
            )
            total_train_t += epoch_t
            epochs_trained += 1

            val_metrics, _ = evaluate(model, val_loader, device)
            val_acc = val_metrics["top1"]

            log = (f"  Epoch {epoch+1}/{num_epochs}  "
                   f"loss={loss:.4f}  t={epoch_t:.1f}s  "
                   f"val_acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc   = val_acc
                patience_count = 0
                best_epoch_num = epoch + 1
                # FIX 5: save best checkpoint
                torch.save({
                    "epoch":            epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "val_acc":          val_acc,
                    "num_labels":       num_labels,
                }, best_ckpt_path)
                print(log + "  ✓ saved")
            else:
                patience_count += 1
                print(log + f"  (patience {patience_count}/{early_stopping_patience})")
                if patience_count >= early_stopping_patience:
                    print(f"  Early stop at epoch {epoch+1} "
                          f"(best: epoch {best_epoch_num})")
                    break

        # FIX 5: reload best checkpoint before test eval
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Reloaded best model "
              f"(epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")

        # FIX 10: capture CPU memory peak
        if is_cpu:
            cpu_snapshot = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        test_metrics, avg_ms = evaluate(model, test_loader, device)
        mem = measure_peak_memory_mb(device, cpu_snapshot)
        params = count_params(model)

        result = BenchmarkResult(
            mode=run_label,
            accuracy_top1=test_metrics["top1"],
            accuracy_top3=test_metrics["top3"],
            accuracy_top5=test_metrics["top5"],
            accuracy_top10=test_metrics["top10"],
            avg_inference_ms=avg_ms,
            peak_memory_mb=mem,
            total_params=params,
            epochs_trained=epochs_trained,
            best_epoch=best_epoch_num,
        )
        results.append(result)

        print(f"\n  Test Results:")
        print(f"    Top-1  Accuracy : {test_metrics['top1']:.4f}")
        print(f"    Top-3  Accuracy : {test_metrics['top3']:.4f}")
        print(f"    Top-5  Accuracy : {test_metrics['top5']:.4f}")
        print(f"    Top-10 Accuracy : {test_metrics['top10']:.4f}")
        print(f"    Avg batch latency: {avg_ms:.2f} ms")
        print(f"    Peak memory      : {mem:.1f} MB")
        print(f"    Train time       : {total_train_t:.1f}s")
        print(f"    Epochs           : {epochs_trained} (best: {best_epoch_num})")

    return results[0], results[1]


# ─────────────────────────────────────────────
# 7. COMPARISON SUMMARY
# ─────────────────────────────────────────────

def print_comparison(baseline: BenchmarkResult, tome: BenchmarkResult):
    speedup = baseline.avg_inference_ms / (tome.avg_inference_ms + 1e-9)

    print("\n" + "=" * 75)
    print("  COMPARISON SUMMARY")
    print("=" * 75)
    print(f"{'Metric':<25} {'Baseline':>14} {'ToMe':>14} {'Δ':>14}")
    print("-" * 75)

    for k, label in [("top1","Top-1"), ("top3","Top-3"),
                     ("top5","Top-5"), ("top10","Top-10")]:
        b = getattr(baseline, f"accuracy_{k}")
        t = getattr(tome,     f"accuracy_{k}")
        print(f"{label+' Accuracy':<25} {b:>14.4f} {t:>14.4f} "
              f"{(t-b)*100:>13.2f}%")

    print("-" * 75)
    print(f"{'Avg Inference (ms)':<25} "
          f"{baseline.avg_inference_ms:>14.2f} "
          f"{tome.avg_inference_ms:>14.2f} "
          f"{tome.avg_inference_ms - baseline.avg_inference_ms:>13.2f}")
    print(f"{'Peak Memory (MB)':<25} "
          f"{baseline.peak_memory_mb:>14.1f} "
          f"{tome.peak_memory_mb:>14.1f} "
          f"{tome.peak_memory_mb - baseline.peak_memory_mb:>13.1f}")
    print(f"{'Best epoch':<25} "
          f"{baseline.best_epoch:>14} "
          f"{tome.best_epoch:>14}")
    print(f"{'Epochs trained':<25} "
          f"{baseline.epochs_trained:>14} "
          f"{tome.epochs_trained:>14}")
    print("-" * 75)
    print(f"  Speed-up factor : {speedup:.2f}×")
    print("=" * 75)


# ─────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    script_start = time.perf_counter()

    train = pd.read_csv("C:\\Users\\Admin\\Downloads\\New folder\\data\\train_set.csv")

    # Sample strategy: top-10 journals × 300 + 300 tail
    top_labels   = train["Label"].value_counts().nlargest(10).index.tolist()
    top_samples  = (train[train["Label"].isin(top_labels)]
                    .groupby("Label")
                    .sample(300, random_state=42))
    tail_samples = (train[~train["Label"].isin(top_labels)]
                    .sample(300, random_state=42))
    df = pd.concat([top_samples, tail_samples]).reset_index(drop=True)

    baseline, tome = run_benchmark(
        df,
        num_epochs=20,
        batch_size=16,
        max_length=128,
        tome_r=8,
        tome_metric="keys",          # try: "keys" | "queries" | "hidden"
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        early_stopping_patience=3,
        save_dir="./checkpoints",
        seed=42,
    )

    print_comparison(baseline, tome)

    total = time.perf_counter() - script_start
    print(f"\n{'='*60}")
    print(f"  Total time: {total:.1f}s  ({total/60:.1f} min)")
    print(f"{'='*60}")