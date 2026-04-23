from torch import Tensor
import torch.nn as nn
import math
import time
from transformers import BertModel
from typing import Callable, Tuple
import torch


TOME_MERGE_TIME = {"total_ms": 0.0, "call_count": 0}


def reset_tome_timer() -> None:
    global TOME_MERGE_TIME
    TOME_MERGE_TIME = {"total_ms": 0.0, "call_count": 0}


def get_tome_timer_stats() -> dict:
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