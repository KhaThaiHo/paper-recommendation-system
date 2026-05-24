from torch import Tensor
import torch.nn as nn
import math
import time
from transformers import BertModel
from typing import Callable, Tuple
import torch


TOME_MERGE_TIME = {"total_s": 0.0, "call_count": 0}


def reset_tome_timer() -> None:
    global TOME_MERGE_TIME
    TOME_MERGE_TIME = {"total_s": 0.0, "call_count": 0}


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

            # Merge residual to T' so sizes match for LayerNorm
            residual = merge_fn(residual)               # (B, T', C)

            # FIX 2: Also merge the attention mask so padding stays masked
            # after token reduction. Without this, the mask is silently
            # dropped (shape mismatch guard below) and padding tokens receive
            # full attention weight throughout training.
            if attention_mask is not None:
                # HuggingFace extended mask: (B, 1, 1, T), values 0 / -10000
                # Reshape to (B, T, 1), apply merge with 'amax' so that a real
                # token (0) merged with a padding token (-10000) stays real (0).
                orig_mask_shape = attention_mask.shape
                mask_1d = attention_mask.view(
                    attention_mask.size(0), -1, 1
                )                                       # (B, T, 1) – squeeze middle dims
                # 'amax' keeps the less-restrictive value (real wins over padding)
                merged_mask = merge_fn(mask_1d, mode="amax")  # (B, T', 1)
                # Restore to (B, 1, 1, T') so the add below works
                attention_mask = merged_mask.view(
                    orig_mask_shape[0],
                    *orig_mask_shape[1:-1],
                    merged_mask.size(1),
                )

            t_merge_end = time.perf_counter()
            TOME_MERGE_TIME["total_s"]  += t_merge_end - t_merge_start
            TOME_MERGE_TIME["call_count"] += 1
        # ────────────────────────────────────────────────────────────────

        scale = math.sqrt(self.attention_head_size)
        scores = torch.matmul(q, k.transpose(-1, -2)) / scale

        if attention_mask is not None and attention_mask.shape[-1] == scores.shape[-1]:
            scores = scores + attention_mask

        probs = nn.functional.softmax(scores, dim=-1)
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
    metric: Tensor,
    r: int,
) -> Tuple[Callable, Callable]:
    """
    Bipartite soft matching from the ToMe paper (Bolya et al., 2022).

    Splits the *body* tokens (everything after CLS) into two sets
    (A = even body indices, B = odd body indices), finds the most similar
    cross-set pairs, and returns merge / unmerge functions.

    FIX 1 – CLS position: the original code protected CLS from being a
    *source* (via node_max[:, 0] = -inf) but still let argsort reorder it.
    Because argsort(descending=True) pushes -inf to the tail, CLS ended up
    somewhere in the middle of the output — not at position 0. BERT's pooler
    reads sequence_output[:, 0], so it was classifying on a random body
    token instead of CLS.

    The fix is to exclude CLS entirely from the A/B partitioning and always
    prepend it at position 0 in the merge output.

    FIX 3 (minor) – SEP protection: SEP tokens (last non-padding token) are
    excluded from being sources by setting their score to -inf. They carry
    sentence-boundary information used during pre-training and are cheap to
    keep since they sit in the B-set (destinations only).

    Returns:
        merge   - callable that reduces token count by r, CLS always at [0]
        unmerge - callable that restores original positions (identity for
                  classification; extend if you need token-level output)
    """
    B, T, _ = metric.shape

    with torch.no_grad():
        metric = metric / (metric.norm(dim=-1, keepdim=True) + 1e-6)

        # FIX 1: exclude CLS (index 0) from the matching entirely
        body = metric[:, 1:, :]                         # (B, T-1, d)

        a = body[..., ::2, :]                           # even body tokens
        b = body[..., 1::2, :]                          # odd body tokens
        scores = a @ b.transpose(-1, -2)                # (B, |A|, |B|)

        node_max, node_idx = scores.max(dim=-1)         # best B-match per A token

        # FIX 3: protect SEP (last body token) from being a source.
        # SEP is at body index T-2 (0-based); in the A-set if (T-2) is even.
        sep_body_idx = T - 2                            # position in body
        sep_a_idx    = sep_body_idx // 2               # position in A-set
        node_max = node_max.clone()
        if sep_body_idx % 2 == 0 and sep_a_idx < node_max.size(1):
            node_max[:, sep_a_idx] = -float("inf")

        a_size = a.size(1)
        r = min(r, a_size - 1)

        sorted_idx = node_max.argsort(dim=-1, descending=True)
        src_idx = sorted_idx[..., :r]
        unm_idx = sorted_idx[..., r:]
        dst_idx = node_idx.gather(dim=-1, index=src_idx)

    def merge(x: Tensor, mode: str = "mean") -> Tensor:
        """
        Merge r token pairs; returns a tensor with T - r tokens.
        CLS is always preserved at position 0.

        mode: 'mean'  – average merged tokens (best for hidden states)
              'amax'  – take the max (best for additive attention masks,
                        where 0 = attend and -10000 = ignore)
        """
        # FIX 1: separate CLS so it is never touched by the matching
        cls  = x[:, 0:1, :]                            # (B, 1, C)
        body = x[:, 1:, :]                             # (B, T-1, C)

        src = body[..., ::2, :]
        dst = body[..., 1::2, :].clone()
        n, _, c = src.shape
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
        elif mode == "amax":
            # Used for attention masks: real token (0) beats padding (-10000)
            dst.scatter_reduce_(
                -2,
                dst_idx.unsqueeze(-1).expand(n, r, c),
                matched_src,
                reduce="amax",
                include_self=True,
            )
        else:
            raise ValueError(f"Unsupported merge mode: {mode!r}")

        unmerged = src.gather(
            dim=-2,
            index=unm_idx.unsqueeze(-1).expand(n, n_unm, c),
        )

        # FIX 1: CLS is always first in the output
        return torch.cat([cls, unmerged, dst], dim=1)

    def unmerge(x: Tensor) -> Tensor:
        # Identity for sequence-level classification (pooler uses CLS only).
        # Extend this if you need token-level output (NER, QA, etc.).
        return x

    return merge, unmerge