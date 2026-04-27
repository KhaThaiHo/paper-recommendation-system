"""
Token Merging (ToMe) for Academic Paper Journal Classification
==============================================================

PIPELINE OVERVIEW:
  1. Preprocess: configurable fields (e.g. Title, Abstract, Keywords, Aims,
                 Categories) joined from paper dataset + optional journal
                 dataset → combined text
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

CHANGES FROM v1:
  - [FIX]  Shallow copy of model state dict replaced with copy.deepcopy()
  - [FIX]  CLS token (index 0 in set A) is now protected from being merged
  - [FIX]  attention_mask is now applied to attention scores (padding fix)
  - [PERF] argsort computed once and sliced (was called twice)
  - [NEW]  --fields  : choose any subset/order of input fields at the CLI
  - [NEW]  --journal_csv : optional journal dataset to contribute Aims &
           Categories; merged into paper splits by Label before text is built

CHANGES FROM v2 (logging + run_mode):
  - [NEW] TeeLogger: mirrors all stdout to a per-session .txt log file
  - [NEW] save_session_log(): writes a structured .json log (config + results)
  - [NEW] Config.run_mode: "baseline" | "tome" | "both"  controls which
          models are trained; missing result is returned as None
  - [NEW] Config.log_dir: directory where session logs are saved
  - [CHG] run_benchmark() gains a run_mode parameter; return value is
          (Optional[BenchmarkResult], Optional[BenchmarkResult]) — the
          missing side is None when run_mode != "both"
  - [CHG] __main__ block: initialises TeeLogger, passes run_mode, guards
          print_comparison, and calls save_session_log at the end

USAGE EXAMPLES:
  # Default behaviour (Title + Abstract + Keywords)
  python tome_bert_classifier.py --train_csv train.csv --val_csv val.csv --test_csv test.csv

  # Add journal Aims and Categories after Keywords
  python tome_bert_classifier.py ... --journal_csv journal.csv \
      --fields Title Abstract Keywords Aims Categories

  # Only Abstract + Aims (minimal input)
  python tome_bert_classifier.py ... --journal_csv journal.csv \
      --fields Abstract Aims

  # Reorder: lead with Keywords
  python tome_bert_classifier.py ... --journal_csv journal.csv \
      --fields Keywords Title Abstract Aims Categories
"""

import os           # [NEW v2] needed for makedirs in TeeLogger + save_session_log
import sys          # [NEW v2] needed for TeeLogger (sys.stdout redirect)
import json         # [NEW v2] needed for save_session_log JSON dump
import datetime     # [NEW v2] needed for session timestamp
import copy
import time
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch import Tensor
from typing import Optional, Tuple, Callable, List
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────
# 0. FIELD CONSTANTS
# ─────────────────────────────────────────────

# Columns available from the paper CSVs (train / val / test)
PAPER_FIELDS: List[str] = ["Title", "Abstract", "Keywords"]

# Columns that come from the journal CSV (joined by Label)
JOURNAL_FIELDS: List[str] = ["Aims", "Categories"]

# All recognised field names (validation helper)
ALL_FIELDS: List[str] = PAPER_FIELDS + JOURNAL_FIELDS

# What the pipeline used before this feature was added
DEFAULT_FIELDS: List[str] = PAPER_FIELDS


# ─────────────────────────────────────────────
# 0b. SESSION LOGGER  [NEW v2]
# ─────────────────────────────────────────────

class TeeLogger:
    """
    [NEW v2] Mirrors every print() call to both the terminal and a .txt log
    file simultaneously by replacing sys.stdout.

    WHY HERE: Placing it right after constants means it is available before
    any other code runs.  Zero changes are needed to existing print()
    statements — every line of output is captured automatically.

    Usage:
        logger = TeeLogger("/kaggle/working/logs/session_20240101_120000.txt")
        sys.stdout = logger
        ...                    # all prints go to terminal AND the file
        logger.close()         # restores sys.stdout
    """

    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # buffering=1 → line-buffered so output appears in the file
        # in real time even if the process crashes mid-run.
        self.log = open(filepath, "w", buffering=1, encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self) -> bool:
        # Always report non-interactive so libraries (e.g. HuggingFace) that
        # call sys.stdout.isatty() don't crash and skip ANSI colour codes,
        # which keeps the log file clean.
        return False

    def close(self):
        """Restore original stdout and close the file handle."""
        sys.stdout = self.terminal
        self.log.close()


# ─────────────────────────────────────────────
# 1. DATA PREPARATION
# ─────────────────────────────────────────────

def merge_journal_info(
    paper_df: pd.DataFrame,
    journal_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join journal-level columns (Aims, Categories) into a paper
    DataFrame by matching on the shared 'Label' column.

    Papers whose label has no matching journal entry will have NaN in
    the new columns, which load_and_preprocess() fills with "".

    Args:
        paper_df:   DataFrame with [Title, Abstract, Keywords, Label, …]
        journal_df: DataFrame with [Label, Aims, Categories, …]

    Returns:
        Merged DataFrame (same row count as paper_df, extra cols appended).
    """
    available_journal_cols = [c for c in JOURNAL_FIELDS if c in journal_df.columns]
    if not available_journal_cols:
        print("[Warning] journal_df has no recognised columns "
              f"({JOURNAL_FIELDS}). Skipping merge.")
        return paper_df

    join_cols = ["Label"] + available_journal_cols
    journal_subset = (
        journal_df[join_cols]
        .drop_duplicates(subset=["Label"])
        .reset_index(drop=True)
    )
    merged = paper_df.merge(journal_subset, on="Label", how="left")

    n_matched = merged[available_journal_cols[0]].notna().sum()
    print(f"  [Journal merge] {n_matched}/{len(merged)} papers matched "
          f"a journal entry. Added columns: {available_journal_cols}")
    return merged


def load_and_preprocess(
    df: pd.DataFrame,
    fields: List[str],
) -> pd.DataFrame:
    """
    Build a single 'text' column from the requested fields, joined by [SEP].

    - Fields not present in df are skipped with a warning.
    - NaN / missing values in any field are replaced with "".
    - Order of fields is preserved exactly as supplied.

    Args:
        df:     Input DataFrame (may include paper + journal columns).
        fields: Ordered list of column names to concatenate.

    Returns:
        Copy of df with a new 'text' column.
    """
    df = df.copy()
    valid_parts: List[pd.Series] = []

    for f in fields:
        if f not in df.columns:
            print(f"  [Warning] Field '{f}' not found in DataFrame — skipping.")
            continue
        df[f] = df[f].fillna("")
        valid_parts.append(df[f])

    if not valid_parts:
        raise ValueError(
            f"None of the requested fields {fields} exist in the DataFrame.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df["text"] = valid_parts[0]
    for part in valid_parts[1:]:
        df["text"] = df["text"] + " [SEP] " + part

    return df


def preprocess_split(
    df: pd.DataFrame,
    split_name: str,
    fields: List[str],
    journal_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Full preprocessing for one data split:
      1. Optionally merge journal info (Aims, Categories) by Label.
      2. Build combined 'text' from the requested fields.
      3. Drop rows missing a Label.

    Args:
        df:          Raw split DataFrame (train / val / test).
        split_name:  Label for logging ("train", "validation", "test").
        fields:      Ordered list of field names to include in 'text'.
        journal_df:  Optional journal DataFrame to left-join on Label.

    Returns:
        Processed DataFrame with a 'text' column and no NaN labels.
    """
    if journal_df is not None:
        df = merge_journal_info(df, journal_df)

    split_df = load_and_preprocess(df, fields)
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
        print(f"[Warning] Dropping {dropped_count} samples in {split_name} "
              f"split with unseen labels.")

    filtered_df = split_df[known_mask].reset_index(drop=True)
    if filtered_df.empty:
        raise ValueError(
            f"{split_name} split has no samples with labels seen in training split."
        )

    encoded_labels = raw_labels[known_mask].map(label_to_id).astype(int).tolist()
    texts = filtered_df["text"].tolist()
    return texts, encoded_labels


class PaperDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding        = "max_length",
            truncation     = True,
            max_length     = self.max_length,
            return_tensors = "pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────
# 2. TOKEN MERGING CORE IMPLEMENTATION
# ─────────────────────────────────────────────

def bipartite_soft_matching(
    metric: Tensor,   # (B, T, C) – token features used for similarity
    r: int,           # number of token pairs to merge per layer
) -> Tuple[Callable, Callable]:
    """
    Bipartite soft matching from the ToMe paper (Bolya et al., 2022).

    Splits tokens into two sets (A = even indices, B = odd indices),
    finds the most similar cross-set pairs, and returns merge /
    unmerge functions.

    Fixes vs. v1:
      - CLS token (A[0]) is protected: its match score is forced to -inf
        so it is always placed in the unmerged set.
      - argsort is computed once and sliced (was called twice).
      - r is capped at (A_size - 1) to guarantee CLS stays alive.

    Returns:
        merge   – callable that reduces token count by r
        unmerge – callable that restores original positions (no-op; stored
                  indices are needed for full reconstruction)
    """
    B, T, _ = metric.shape

    with torch.no_grad():
        metric = metric / (metric.norm(dim=-1, keepdim=True) + 1e-6)

        # Set A = even-indexed tokens, Set B = odd-indexed tokens
        a, b = metric[..., ::2, :], metric[..., 1::2, :]   # (B, ≥T/2, C)
        scores = a @ b.transpose(-1, -2)                    # (B, |A|, |B|)

        # Best B-match and its index for every A token
        node_max, node_idx = scores.max(dim=-1)             # (B, |A|)

        # ── [FIX] CLS protection ─────────────────────────────────────────
        # The CLS token sits at sequence position 0 (even → A[0]).
        # Forcing its score to -inf guarantees it lands in unm_idx and
        # is never consumed by another token, preserving the classifier head.
        node_max = node_max.clone()
        node_max[:, 0] = -float("inf")
        # ─────────────────────────────────────────────────────────────────

        # Cap r: can merge at most |A|-1 pairs (CLS slot is reserved)
        a_size = a.size(1)
        r = min(r, a_size - 1)

        # [PERF] Single argsort, then slice — was called twice before
        sorted_idx = node_max.argsort(dim=-1, descending=True)   # (B, |A|)
        src_idx  = sorted_idx[..., :r]        # A tokens to merge  (B, r)
        unm_idx  = sorted_idx[..., r:]        # A tokens to keep   (B, |A|-r)
        dst_idx  = node_idx.gather(dim=-1, index=src_idx)   # matched B tokens

    def merge(x: Tensor, mode: str = "mean") -> Tensor:
        """
        Merge r token pairs; returns a tensor with (T - r) tokens.

        src (A tokens) are gathered, and the r selected ones are
        scatter-averaged into their matched dst (B) tokens.
        Unselected A tokens are concatenated with the updated B set.
        """
        src, dst = x[..., ::2, :], x[..., 1::2, :].clone()
        n, t1, c = src.shape
        n_unm = unm_idx.size(-1)

        matched_src = src.gather(
            dim=-2,
            index=src_idx.unsqueeze(-1).expand(n, r, c),
        )
        if mode == "mean":
            dst.scatter_reduce_(
                -2,
                dst_idx.unsqueeze(-1).expand(n, r, c),
                matched_src,
                reduce="mean",
                include_self=True,
            )
        unmerged = src.gather(
            dim=-2,
            index=unm_idx.unsqueeze(-1).expand(n, n_unm, c),
        )
        # Unmerged A (CLS always first here) || updated B
        return torch.cat([unmerged, dst], dim=1)

    def unmerge(x: Tensor) -> Tensor:
        """Approximately restore original positions (no-op without stored idx)."""
        return x

    return merge, unmerge


# ─────────────────────────────────────────────
# 3. ToMe-PATCHED BERT SELF-ATTENTION
# ─────────────────────────────────────────────

# Global timing accumulator for ToMe merge operations
TOME_MERGE_TIME = {"total_s": 0.0, "call_count": 0}

def reset_tome_timer():
    """Reset the ToMe merge timing counters."""
    global TOME_MERGE_TIME
    TOME_MERGE_TIME = {"total_s": 0.0, "call_count": 0}

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

    Fixes vs. v1:
      - attention_mask is now applied to scores (padding tokens were
        previously attended to without masking).
      - CLS protection is now enforced inside bipartite_soft_matching().
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
        """(B, H, T, d) → merge along T → (B, H, T', d)"""
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

            metric = k.mean(dim=1)                        # (B, T, d)
            merge_fn, _ = bipartite_soft_matching(metric, self.r)

            q = self._merge_heads(q, merge_fn)            # (B, H, T', d)
            k = self._merge_heads(k, merge_fn)
            v = self._merge_heads(v, merge_fn)
            residual = merge_fn(residual)                 # (B, T', C)

            t_merge_end = time.perf_counter()
            TOME_MERGE_TIME["total_s"] += t_merge_end - t_merge_start
            TOME_MERGE_TIME["call_count"] += 1
        # ──────────────────────────────────────────────────────────────────

        scale  = math.sqrt(self.attention_head_size)
        scores = torch.matmul(q, k.transpose(-1, -2)) / scale

        # ── [FIX] Apply attention mask to prevent attending to padding ─────
        # HuggingFace BertModel extends the mask to (B, 1, 1, T) with
        # 0.0 for real tokens and -10000.0 for padding before passing it
        # to the attention layers.  After token merging T becomes T', so
        # we only apply the mask when the sequence length still matches.
        if attention_mask is not None and attention_mask.shape[-1] == scores.shape[-1]:
            scores = scores + attention_mask
        # ──────────────────────────────────────────────────────────────────

        probs = nn.functional.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        ctx   = torch.matmul(probs, v)                    # (B, H, T', d)

        ctx = ctx.permute(0, 2, 1, 3).contiguous()
        ctx = ctx.view(ctx.size(0), ctx.size(1), self.all_head_size)

        # BertSelfOutput: dense → dropout → LayerNorm(x + residual)
        ctx = self.out_dense(ctx)
        ctx = self.out_dropout(ctx)
        attention_output = self.out_LayerNorm(ctx + residual)   # both T'

        attn_weights = probs if output_attentions else None
        return (attention_output, attn_weights)


def patch_bert_with_tome(model: AutoModel, r: int = 8) -> AutoModel:
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

MODEL_NAME = "dmis-lab/biobert-v1.1"   # [CHG v4] single place to change model


class BertClassifier(nn.Module):
    def __init__(self, pretrained_model: AutoModel,
                 num_labels: int, use_tome: bool = False, tome_r: int = 8):
        super().__init__()
        self.bert = copy.deepcopy(pretrained_model)   # independent weights, no download

        if use_tome:
            self.bert = patch_bert_with_tome(self.bert, r=tome_r)
            print(f"[ToMe ON]  Merging {tome_r} token pairs per layer")
        else:
            print("[ToMe OFF] Standard BERT (no merging)")

        hidden_size = self.bert.config.hidden_size   # 768
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        # CLS token representation
        # input_ids:       (B, T)
        # last_hidden_state: (B, T', 768)  [T' ≤ T when ToMe is on]
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # (B, 768)
        return self.classifier(cls)             # (B, num_labels)


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
    avg_inference_s: float
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
    _, topk_preds = torch.topk(all_logits, min(k, all_logits.size(1)), dim=-1)
    topk_preds = topk_preds.cpu().numpy()
    correct = sum(
        1 for true_label, pred_k in zip(all_labels, topk_preds)
        if true_label in pred_k
    )
    return correct / len(all_labels)


def evaluate(model, loader, device) -> Tuple[dict, float]:
    model.eval()
    all_logits = []
    all_labels = []
    latencies  = []

    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"]

            t0 = time.perf_counter()
            logits = model(ids, mask)
            latencies.append(time.perf_counter() - t0)

            all_logits.append(logits.cpu())
            all_labels.extend(lbls.tolist())

    all_logits = torch.cat(all_logits, dim=0)
    metrics = {
        "top1":  compute_topk_accuracy(all_logits, all_labels, 1),
        "top3":  compute_topk_accuracy(all_logits, all_labels, 3),
        "top5":  compute_topk_accuracy(all_logits, all_labels, 5),
        "top10": compute_topk_accuracy(all_logits, all_labels, 10),
    }
    return metrics, float(np.mean(latencies))


def peak_memory_mb(device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e6
    return 0.0


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# 5b. SESSION LOG WRITER  [NEW v2]
# ─────────────────────────────────────────────

def save_session_log(
    config,
    baseline: Optional["BenchmarkResult"],
    tome: Optional["BenchmarkResult"],
    total_time_s: float,
    log_dir: str,
    session_id: str,
) -> str:
    """
    [NEW v2] Write a structured JSON file capturing config + results for
    this training session.

    WHY A SEPARATE FUNCTION: Keeps all serialisation logic in one place;
    run_benchmark() and __main__ stay unchanged except for one call site.

    File path: <log_dir>/session_<session_id>.json
                e.g.  /kaggle/working/logs/session_20240101_120000.json

    Args:
        config       : the Config class (attributes read directly).
        baseline     : BenchmarkResult for ToMe-OFF run, or None.
        tome         : BenchmarkResult for ToMe-ON run, or None.
        total_time_s : wall-clock seconds for the entire __main__ block.
        log_dir      : directory to write the file into.
        session_id   : timestamp string used as filename suffix.

    Returns:
        Absolute path of the written JSON file.
    """
    def _result_to_dict(r: Optional["BenchmarkResult"]) -> Optional[dict]:
        if r is None:
            return None
        return {
            "mode":             r.mode.strip(),
            "accuracy_top1":   r.accuracy_top1,
            "accuracy_top3":   r.accuracy_top3,
            "accuracy_top5":   r.accuracy_top5,
            "accuracy_top10":  r.accuracy_top10,
            "avg_inference_s": r.avg_inference_s,
            "peak_memory_mb":  r.peak_memory_mb,
            "total_params":    r.total_params,
            "epochs_trained":  r.epochs_trained,
        }

    log_data = {
        "session_id":   session_id,
        "timestamp":    session_id,          # human-readable duplicate
        "config": {
            "run_mode":                 config.run_mode,
            "fields":                   config.fields,
            "num_epochs":               config.num_epochs,
            "batch_size":               config.batch_size,
            "max_length":               config.max_length,
            "tome_r":                   config.tome_r,
            "learning_rate":            config.learning_rate,
            "early_stopping_patience":  config.early_stopping_patience,
            "train_csv":                config.train_csv,
            "val_csv":                  config.val_csv,
            "test_csv":                 config.test_csv,
            "journal_csv":              getattr(config, "journal_csv", None),
        },
        "results": {
            "baseline": _result_to_dict(baseline),
            "tome":     _result_to_dict(tome),
        },
        "total_time_s": round(total_time_s, 3),
    }

    os.makedirs(log_dir, exist_ok=True)
    json_path = os.path.join(log_dir, f"session_{session_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

    print(f"\n[Log saved → JSON] {json_path}")
    return json_path


# ─────────────────────────────────────────────
# 6. FULL BENCHMARK PIPELINE
# ─────────────────────────────────────────────

def run_benchmark(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    journal_df: Optional[pd.DataFrame] = None,
    fields: Optional[List[str]] = None,
    num_epochs: int = 10,
    batch_size: int = 16,
    max_length: int = 256,
    tome_r: int = 8,
    learning_rate: float = 2e-5,
    early_stopping_patience: int = 3,
    run_mode: str = "both",                    # [NEW v2] "baseline" | "tome" | "both"
) -> Tuple[Optional[BenchmarkResult], Optional[BenchmarkResult]]:
    """
    Trains and evaluates one or both models depending on run_mode:
      - "baseline" → trains BERT only;        returns (baseline, None)
      - "tome"     → trains BERT + ToMe only; returns (None, tome)
      - "both"     → trains both;             returns (baseline, tome)

    Args:
        train_df / val_df / test_df:
            Paper DataFrames with columns [Title, Abstract, Keywords, Label].
        journal_df:
            Optional journal DataFrame with [Label, Aims, Categories].
            When provided, Aims / Categories are left-joined into each
            paper split by Label before building the 'text' field.
        fields:
            Ordered list of column names to concatenate into 'text'.
            Defaults to DEFAULT_FIELDS = ["Title", "Abstract", "Keywords"].
            Any column present after the optional journal merge is valid:
              Title, Abstract, Keywords, Aims, Categories
        run_mode:                                   [NEW v2]
            Which model(s) to train and evaluate.
            "baseline" skips ToMe; "tome" skips baseline; "both" runs both.
        num_epochs / batch_size / max_length / tome_r / learning_rate /
        early_stopping_patience: training hyper-parameters.

    Returns:
        (baseline_result, tome_result) – either may be None depending on
        run_mode.
    """
    if fields is None:
        fields = DEFAULT_FIELDS

    # Validate field names
    unknown = [f for f in fields if f not in ALL_FIELDS]
    if unknown:
        raise ValueError(
            f"Unknown field(s): {unknown}. "
            f"Valid options are: {ALL_FIELDS}"
        )

    # ── [NEW v2] Build the list of use_tome flags to iterate ────────────
    # Map run_mode → the booleans passed to BertClassifier.
    # "baseline" → [False], "tome" → [True], "both" → [False, True]
    _mode_map = {
        "baseline": [False],
        "tome":     [True],
        "both":     [False, True],
    }
    if run_mode not in _mode_map:
        raise ValueError(
            f"Invalid run_mode='{run_mode}'. "
            f"Choose from: {list(_mode_map.keys())}"
        )
    modes_to_run = _mode_map[run_mode]
    # ─────────────────────────────────────────────────────────────────────

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print(f"Input fields : {fields}")
    print(f"Run mode     : {run_mode}")          # [NEW v2] log active mode
    if journal_df is not None:
        print(f"Journal CSV  : provided ({len(journal_df)} rows)")
    print("=" * 55)

    # ── Preprocess ──────────────────────────────────────────────────────
    t_preprocess = time.perf_counter()
    print(f"Starting preprocess train set...")
    train_df = preprocess_split(train_df, "train",      fields, journal_df)
    print(f"Trainset preprocessed completedly")

    print(f"Starting preprocess validation set...")
    val_df   = preprocess_split(val_df,   "validation", fields, journal_df)
    print(f"Validation set preprocessed completedly")

    print(f"Starting preprocess testset...")
    test_df  = preprocess_split(test_df,  "test",       fields, journal_df)
    print(f"Test set preprocessed completedly")

    le = LabelEncoder()
    le.fit(train_df["Label"].astype(str))
    label_to_id = {label: idx for idx, label in enumerate(le.classes_)}

    print(f"Start encoding and spliting...")
    X_train, y_train = encode_split_labels(train_df, label_to_id, "train")
    X_val,   y_val   = encode_split_labels(val_df,   label_to_id, "validation")
    X_test,  y_test  = encode_split_labels(test_df,  label_to_id, "test")
    print(f"Encoding and splitting completed.")

    num_labels = len(le.classes_)
    print(f"Classes: {num_labels} | Train: {len(y_train)} "
          f"| Val: {len(y_val)} | Test: {len(y_test)}")

    t_tok = time.perf_counter()
    print(f"Downloading tokenizer and base model: {MODEL_NAME} …")
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModel.from_pretrained(MODEL_NAME)
    print(f"  [Loaded tokenizer + base model]: {time.perf_counter() - t_tok:.2f}s")

    # ── Datasets & Loaders ──────────────────────────────────────────────
    t_ds = time.perf_counter()
    print(f"Starting tokenize train set...")
    train_ds = PaperDataset(X_train, y_train, tokenizer, max_length)
    print(f"Tokenize train set completed.")

    print(f"Starting tokenize validation set...")
    val_ds   = PaperDataset(X_val,   y_val,   tokenizer, max_length)
    print(f"Tokenize validation set completed.")

    print(f"Starting tokenize test set...")
    test_ds  = PaperDataset(X_test,  y_test,  tokenizer, max_length)
    print(f"Tokenize test set completed.")

    print(f"  [Created datasets]: {time.perf_counter() - t_ds:.2f}s")

    t_dl = time.perf_counter()
    print(f"Starting to load datasets to DataLoader with batch_size: {batch_size}...")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)
    print(f"  [Created dataloaders]: {time.perf_counter() - t_dl:.2f}s")
    print(f"  Total preprocessing: {time.perf_counter() - t_preprocess:.2f}s\n")

    # ── [CHG v2] Iterate only over modes_to_run (was hard-coded [False, True])
    results = []
    for use_tome in modes_to_run:
        if results:
            torch.cuda.empty_cache()

        label = "ToMe ON " if use_tome else "ToMe OFF"
        print(f"\n── {label} ─────────────────────────────────────────")

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        t_init = time.perf_counter()
        model = BertClassifier(
            pretrained_model = base_model,   # [CHG v4] pre-loaded, no download
            num_labels       = num_labels,
            use_tome         = use_tome,
            tome_r           = tome_r,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        print(f"  [Model initialized]: {time.perf_counter() - t_init:.2f}s")

        best_val_acc    = 0.0
        patience_counter = 0
        best_epoch       = 0
        epochs_trained   = 0
        best_model_state = None
        total_train_time = 0.0

        for epoch in range(num_epochs):
            reset_tome_timer()
            print(f"Starting training epoch {epoch+1}...")
            loss, epoch_time = train_one_epoch(model, train_loader, optimizer, criterion, device)
            print(f"Training epoch {epoch + 1} completed.")
            total_train_time += epoch_time
            epochs_trained   += 1

            t_eval = time.perf_counter()
            print(f"Starting evaluation on vadilation...")
            val_metrics, _ = evaluate(model, val_loader, device)
            eval_time_s = time.perf_counter() - t_eval
            val_acc = val_metrics["top1"]
            print(f"Evaluation on validation completed.")

            tome_stats = get_tome_timer_stats()
            tome_info  = (
                f"  ToMe merge: {tome_stats['total_s']:.2f}s "
                f"({tome_stats['call_count']} calls)"
                if tome_stats["call_count"] > 0 else ""
            )
            print(
                f"  Epoch {epoch+1}/{num_epochs}  loss={loss:.4f}  "
                f"train={epoch_time:.2f}s  eval={eval_time_s:.2f}s  "
                f"val_acc={val_acc:.4f}{tome_info}",
                end="",
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_epoch = epoch + 1
                # [FIX] deepcopy instead of shallow .copy() — tensors are now independent
                best_model_state = copy.deepcopy(model.state_dict())
                print(" ✓")
            else:
                patience_counter += 1
                print(f" (patience: {patience_counter}/{early_stopping_patience})")
                if patience_counter >= early_stopping_patience:
                    print(f"  Early stopping at epoch {epoch+1} "
                          f"(best epoch: {best_epoch})")
                    break

        print(f"  [Training completed]: {total_train_time:.2f}s ({epochs_trained} epochs)")

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        t_test = time.perf_counter()
        test_metrics, avg_s = evaluate(model, test_loader, device)
        print(f"  [Testing completed]: {time.perf_counter() - t_test:.2f}s")

        mem    = peak_memory_mb(device)
        params = count_params(model)

        result = BenchmarkResult(
            mode            = label,
            accuracy_top1   = test_metrics["top1"],
            accuracy_top3   = test_metrics["top3"],
            accuracy_top5   = test_metrics["top5"],
            accuracy_top10  = test_metrics["top10"],
            avg_inference_s = avg_s,
            peak_memory_mb  = mem,
            total_params    = params,
            epochs_trained  = epochs_trained,
        )
        results.append(result)

        print(f"\n  Test Results:")
        print(f"    Top-1  Accuracy: {test_metrics['top1']:.4f}")
        print(f"    Top-3  Accuracy: {test_metrics['top3']:.4f}")
        print(f"    Top-5  Accuracy: {test_metrics['top5']:.4f}")
        print(f"    Top-10 Accuracy: {test_metrics['top10']:.4f}")
        print(f"    Avg batch inference: {avg_s:.2f} s")
        print(f"    Peak GPU memory:     {mem:.1f} MB")
        print(f"    Total training time: {total_train_time:.2f}s")
        print(f"    Epochs trained:      {epochs_trained}/{num_epochs}")

    # ── [CHG v2] Return (Optional, Optional) instead of results[0], results[1]
    # Look up each result by its mode label so the caller always gets
    # (baseline, tome) in the right slots regardless of run_mode order.
    baseline_result = next(
        (r for r in results if r.mode.strip() == "ToMe OFF"), None
    )
    tome_result = next(
        (r for r in results if r.mode.strip() == "ToMe ON"), None
    )
    return baseline_result, tome_result


# ─────────────────────────────────────────────
# 7. RESULTS DISPLAY
# ─────────────────────────────────────────────

def print_comparison(baseline: BenchmarkResult, tome: BenchmarkResult):
    speedup = baseline.avg_inference_s / (tome.avg_inference_s + 1e-9)

    print("\n" + "=" * 75)
    print("  COMPARISON SUMMARY – TOP-K ACCURACY TABLE")
    print("=" * 75)
    print(f"{'Metric':<25} {'Baseline':>15} {'ToMe':>15} {'Δ':>15}")
    print("-" * 75)

    for k, attr in [(1, "accuracy_top1"), (3, "accuracy_top3"),
                    (5, "accuracy_top5"), (10, "accuracy_top10")]:
        b_val = getattr(baseline, attr)
        t_val = getattr(tome, attr)
        delta = (t_val - b_val) * 100
        print(f"{'Top-' + str(k) + ' Accuracy':<25} {b_val:>15.4f} {t_val:>15.4f} {delta:>14.2f}%")

    print("-" * 75)
    print(f"{'Avg Inference (s)':<25} {baseline.avg_inference_s:>15.2f} "
          f"{tome.avg_inference_s:>15.2f} "
          f"{(tome.avg_inference_s - baseline.avg_inference_s):>14.2f}")
    print(f"{'Peak GPU Memory (MB)':<25} {baseline.peak_memory_mb:>15.1f} "
          f"{tome.peak_memory_mb:>15.1f} "
          f"{(tome.peak_memory_mb - baseline.peak_memory_mb):>14.1f}")
    print(f"{'Epochs Trained':<25} {baseline.epochs_trained:>15} "
          f"{tome.epochs_trained:>15}")
    print("-" * 75)
    print(f"  Speed-up factor: {speedup:.2f}×")
    print("=" * 75)


# ─────────────────────────────────────────────
# 8. ENTRY POINT  [CHG v2]
# ─────────────────────────────────────────────

class Config:
    train_csv   = "/kaggle/input/datasets/basdong/paper-submission/train_set.csv"
    val_csv     = "/kaggle/input/datasets/basdong/paper-submission/val_set.csv"
    test_csv    = "/kaggle/input/datasets/basdong/paper-submission/test_set.csv"
    journal_csv = "/kaggle/input/datasets/basdong/paper-submission/journal_category.csv"

    fields = ["Title", "Abstract", "Keywords", "Aims"]  # Manually set your list

    # ── [NEW v2] run_mode: "baseline" | "tome" | "both" ─────────────────
    run_mode = "both"

    # ── [NEW v2] Where session logs (.txt + .json) are written ──────────
    log_dir  = "/kaggle/working/logs"

    num_epochs               = 10
    batch_size               = 8
    max_length               = 512
    tome_r                   = 8
    learning_rate            = 2e-5
    early_stopping_patience  = 3


if __name__ == "__main__":
    # ── [NEW v2] Create a unique session ID from the current timestamp ───
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── [NEW v2] Start TeeLogger — all prints below go to terminal + file
    log_txt_path = os.path.join(Config.log_dir, f"session_{session_id}.txt")
    logger = TeeLogger(log_txt_path)
    sys.stdout = logger
    print(f"[Session {session_id}] Log file: {log_txt_path}")

    try:
        script_start = time.perf_counter()

        # Load Data
        train_df   = pd.read_csv(Config.train_csv)
        val_df     = pd.read_csv(Config.val_csv)
        test_df    = pd.read_csv(Config.test_csv)
        journal_df = pd.read_csv(Config.journal_csv)

        # Run Benchmark — pass run_mode so only requested models train
        baseline, tome = run_benchmark(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            journal_df=journal_df,
            fields=Config.fields,
            num_epochs=Config.num_epochs,
            batch_size=Config.batch_size,
            max_length=Config.max_length,
            tome_r=Config.tome_r,
            learning_rate=Config.learning_rate,
            early_stopping_patience=Config.early_stopping_patience,
            run_mode=Config.run_mode,              # [NEW v2]
        )

        # ── [NEW v2] Only print comparison when both results exist ───────
        if baseline is not None and tome is not None:
            print_comparison(baseline, tome)

        total_time = time.perf_counter() - script_start
        print(f"\n{'='*55}")
        print(f" TOTAL SCRIPT EXECUTION TIME: {total_time:.2f}s")
        print(f"{'='*55}")

        # ── [NEW v2] Save structured JSON log ────────────────────────────
        save_session_log(
            config=Config,
            baseline=baseline,
            tome=tome,
            total_time_s=total_time,
            log_dir=Config.log_dir,
            session_id=session_id,
        )
        print(f"[Log saved → TXT]  {log_txt_path}")

    finally:
        # Always restore stdout even if an exception occurred
        logger.close()