import math
import time
from typing import Callable, Tuple

import torch
from torch import Tensor
import torch.nn as nn
from transformers import BertModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


TOME_MERGE_TIME = {"total_s": 0.0, "call_count": 0}


def reset_tome_timer() -> None:
    global TOME_MERGE_TIME
    TOME_MERGE_TIME = {"total_s": 0.0, "call_count": 0}


def get_tome_timer_stats() -> dict:
    return TOME_MERGE_TIME.copy()


def _to_dense_contiguous(x: Tensor) -> Tensor:
    """
    Guarantee a dense, contiguous (strided) tensor.

    Why two checks?
      - x.is_sparse     → True  only for sparse_coo (legacy flag).
      - x.layout        → catches sparse_coo, sparse_csr, sparse_bsr, jagged,
                          and any future non-strided layout.
    In some Kaggle/PyTorch builds one of the two may behave unexpectedly, so
    both are tested.  For a normal strided tensor both are False/strided and
    the branch is skipped with no overhead.
    """
    if x.is_sparse or (x.layout != torch.strided):
        x = x.to_dense()
    if not x.is_contiguous():
        x = x.contiguous()
    return x


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

        self.last_attention_mask = None

    def _transpose(self, x: Tensor) -> Tensor:
        """Project (B, T, C) → (B, H, T, d) with full sparse-safety."""
        # ── Step 1: guarantee a dense, contiguous tensor ─────────────────
        x = _to_dense_contiguous(x)

        # ── Step 2: hard guard — gives a clear diagnostic instead of the
        #    cryptic "permute(sparse_coo)…" message if densification failed.
        if x.layout != torch.strided or x.is_sparse:
            raise RuntimeError(
                f"_transpose: _to_dense_contiguous did not produce a strided "
                f"tensor. layout={x.layout}, is_sparse={x.is_sparse}, "
                f"shape={tuple(x.shape)}.  "
                f"This is a PyTorch internals issue; please report the exact "
                f"PyTorch version."
            )

        # ── Step 3: shape guard ───────────────────────────────────────────
        if x.dim() != 3:
            raise RuntimeError(
                f"_transpose expected a 3-D tensor (B, T, C) but got "
                f"{x.dim()}-D with shape {tuple(x.shape)}"
            )

        # ── Step 4: reshape + permute using view (requires contiguous) ────
        # view() raises immediately on non-contiguous or sparse input, unlike
        # reshape() which can silently return the original tensor for sparse.
        B, T, _ = x.shape
        return (
            x.view(B, T, self.num_attention_heads, self.attention_head_size)
            .permute(0, 2, 1, 3)
        )

    @staticmethod
    def _merge_heads(x: Tensor, merge_fn: Callable) -> Tensor:
        """(B, H, T, d) -> merge along T -> (B, H, T', d)"""
        B, H, T, d = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, H * d)
        x = merge_fn(x)
        T2 = x.size(1)
        return x.view(B, T2, H, d).permute(0, 2, 1, 3)

    @staticmethod
    def _merge_attention_mask(attention_mask: Tensor, merge_fn: Callable) -> Tensor:
        """Merge the attention mask to match merged token length."""
        if attention_mask.dim() == 4:
            # Extended mask from get_extended_attention_mask:
            #   0.0 = valid token, -10000.0 = padding.
            mask = attention_mask[:, 0, 0, :]           # (B, T)
            binary_mask = (mask == 0).float()            # 1.0 = valid, 0.0 = padding
        elif attention_mask.dim() == 2:
            # Raw binary mask: 1 = valid, 0 = padding.
            mask = attention_mask
            binary_mask = mask.float()                   # 1.0 = valid, 0.0 = padding
        else:
            return attention_mask

        # Keep a position valid (1.0) if ANY merged token was valid.
        merged = merge_fn(binary_mask.unsqueeze(-1), mode="max").squeeze(-1)

        # Convert back: valid → 0.0, padding → -10000.0
        extended = (1.0 - merged) * -10000.0
        return extended[:, None, None, :]

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
        # ── Dense guard at the entry point ────────────────────────────────
        # This is the primary fix.  hidden_states must be dense before:
        #   (a) the Q/K/V linear projections  — nn.Linear propagates sparsity
        #   (b) the residual assignment       — used for skip-connection
        #   (c) bipartite_soft_matching       — needs a dense metric tensor
        #
        # Sources of non-strided hidden_states observed in practice:
        #   • DataParallel scatter producing sparse-layout shards
        #   • gather/scatter_reduce_ in the merge function returning non-strided
        #     views under certain PyTorch builds
        # ──────────────────────────────────────────────────────────────────
        hidden_states = _to_dense_contiguous(hidden_states)

        residual = hidden_states                         # (B, T, C) dense

        q = self._transpose(self.query(hidden_states))   # (B, H, T, d)
        k = self._transpose(self.key(hidden_states))
        v = self._transpose(self.value(hidden_states))

        current_attention_mask = attention_mask

        # ── Token Merging ─────────────────────────────────────────────────
        T = hidden_states.size(1)
        if self.r > 0 and T > 2:
            global TOME_MERGE_TIME

            t0 = time.perf_counter()

            merge_fn, _ = bipartite_soft_matching(hidden_states, self.r)

            q = self._merge_heads(q, merge_fn)           # (B, H, T', d)
            k = self._merge_heads(k, merge_fn)
            v = self._merge_heads(v, merge_fn)

            residual = merge_fn(residual).contiguous()   # (B, T', C)

            if attention_mask is not None:
                current_attention_mask = self._merge_attention_mask(
                    attention_mask, merge_fn
                )

            TOME_MERGE_TIME["total_s"] += time.perf_counter() - t0
            TOME_MERGE_TIME["call_count"] += 1
        # ──────────────────────────────────────────────────────────────────

        scale = math.sqrt(self.attention_head_size)
        scores = torch.matmul(q, k.transpose(-1, -2)) / scale

        if current_attention_mask is not None:
            if current_attention_mask.shape[-1] != scores.shape[-1]:
                raise RuntimeError(
                    f"Attention mask length {current_attention_mask.shape[-1]} "
                    f"does not match score length {scores.shape[-1]}"
                )
            scores = scores + current_attention_mask

        probs = nn.functional.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        if head_mask is not None:
            probs = probs * head_mask
        ctx = torch.matmul(probs, v)                     # (B, H, T', d)

        ctx = ctx.permute(0, 2, 1, 3).contiguous()
        ctx = ctx.view(ctx.size(0), ctx.size(1), self.all_head_size)

        # BertSelfOutput: dense → dropout → LayerNorm(x + residual)
        ctx = self.out_dense(ctx)
        ctx = self.out_dropout(ctx)
        attention_output = self.out_LayerNorm(ctx + residual)  # both T'

        self.last_attention_mask = current_attention_mask

        if output_attentions:
            return (attention_output, probs)
        return (attention_output, None)


class ToMeBertEncoder(nn.Module):
    """BertEncoder wrapper that carries ToMe's reduced mask layer by layer."""

    def __init__(self, original_encoder):
        super().__init__()
        self.config = original_encoder.config
        self.layer = original_encoder.layer
        self.gradient_checkpointing = getattr(
            original_encoder, "gradient_checkpointing", False
        )
        if hasattr(original_encoder, "_gradient_checkpointing_func"):
            self._gradient_checkpointing_func = (
                original_encoder._gradient_checkpointing_func
            )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        **kwargs,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        current_attention_mask = attention_mask

        for i, layer_module in enumerate(self.layer):

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=current_attention_mask,
                head_mask=layer_head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            # Propagate the (possibly reduced) mask to the next layer.
            current_attention_mask = getattr(
                layer_module.attention,
                "last_attention_mask",
                current_attention_mask,
            )

            if output_attentions:
                # layer_outputs is exactly (hidden_states, attn_probs) — 2 elements.
                all_self_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


def patch_bert_with_tome(model: BertModel, r: int = 8) -> BertModel:
    """
    Replace every BertAttention block with ToMeBertAttention.
    Patching the full block (not just self-attn) is required so that
    the residual connection is also merged to match the reduced T'.
    """
    if isinstance(r, int):
        r = [r] * len(model.encoder.layer)

    assert len(r) == len(model.encoder.layer)
    for i, layer in enumerate(model.encoder.layer):
        layer.attention = ToMeBertAttention(layer.attention, r=r[i])
    model.encoder = ToMeBertEncoder(model.encoder)
    return model


def bipartite_soft_matching(
    metric: Tensor,
    r: int,
) -> Tuple[Callable, Callable]:
    """
    Bipartite soft matching from the ToMe paper (Bolya et al., 2022).

    Splits tokens into two sets (A = even indices, B = odd indices),
    finds the most similar cross-set pairs, and returns merge / unmerge
    functions.

    Returns:
        merge   - callable that reduces token count by r
        unmerge - callable that restores original positions (no-op)
    """
    B, T, C = metric.shape
    if T <= 2 or r <= 0:
        def identity(x, mode: str = "mean") -> Tensor:
            return x
        return identity, identity

    with torch.no_grad():
        metric = metric / (metric.norm(dim=-1, keepdim=True) + 1e-6)

        metric_wo_cls = metric[:, 1:, :]
        a = metric_wo_cls[..., ::2, :]
        b = metric_wo_cls[..., 1::2, :]

        if b.size(1) == 0:
            def identity(x, mode: str = "mean") -> Tensor:
                return x
            return identity, identity

        scores = torch.matmul(a.float(), b.float().transpose(-1, -2))

        node_max, node_idx = scores.max(dim=-1)
        a_size = a.size(1)

        r = min(r, a_size - 1)

        if r <= 0:
            def identity(x, mode: str = "mean") -> Tensor:
                return x
            return identity, identity

        node_max = node_max.clone()
        sorted_idx = node_max.argsort(dim=-1, descending=True)
        src_idx = sorted_idx[..., :r]
        unm_idx = sorted_idx[..., r:]
        dst_idx = node_idx.gather(dim=-1, index=src_idx)

    def merge(x: Tensor, mode: str = "mean") -> Tensor:
        """Merge r token pairs; returns a tensor with T - r tokens."""
        # Ensure dense input before any index / scatter operations.
        x = _to_dense_contiguous(x)

        cls_x    = x[:, :1, :]
        x_wo_cls = x[:, 1:, :]

        src = x_wo_cls[..., ::2, :].contiguous()
        dst = x_wo_cls[..., 1::2, :].clone()

        n, _, c = src.shape
        n_unm   = unm_idx.size(-1)

        matched_src = src.gather(
            dim=-2,
            index=src_idx.unsqueeze(-1).expand(n, r, c),
        )
        reduce_mode = "mean" if mode == "mean" else "amax"
        dst.scatter_reduce_(
            dim=-2,
            index=dst_idx.unsqueeze(-1).expand(n, r, c),
            src=matched_src,
            reduce=reduce_mode,
            include_self=True,
        )

        unmerged = src.gather(
            dim=-2,
            index=unm_idx.unsqueeze(-1).expand(n, n_unm, c),
        )

        pos_a = unm_idx * 2
        pos_b = (
            torch.arange(dst.size(1), device=x.device)
            .unsqueeze(0)
            .expand(n, -1) * 2 + 1
        )

        out_tokens = torch.cat([unmerged, dst], dim=1)
        out_pos    = torch.cat([pos_a,    pos_b], dim=1)

        order  = out_pos.argsort(dim=-1)
        merged = out_tokens.gather(
            dim=1,
            index=order.unsqueeze(-1).expand(n, out_tokens.size(1), c),
        )

        out = torch.cat([cls_x, merged], dim=1)
        # Guarantee strided output: gather/scatter can produce non-strided
        # tensors in certain PyTorch builds.
        return _to_dense_contiguous(out)

    def unmerge(x: Tensor) -> Tensor:
        return x

    return merge, unmerge