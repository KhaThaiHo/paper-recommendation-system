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

    @staticmethod
    def _merge_attention_mask(attention_mask: Tensor, merge_fn: Callable) -> Tensor:
        """Merge the attention mask to match merged token length."""
        if attention_mask.dim() == 4:
            if attention_mask.size(2) == 1:
                mask = attention_mask[:, 0, 0, :]  # (B, T)
            else:
                mask = attention_mask[:, 0, 0, :]  # (B, T), key mask is shared by queries
        elif attention_mask.dim() == 2:
            mask = attention_mask
        else:
            return attention_mask

        merged = merge_fn(mask.unsqueeze(-1), mode="max").squeeze(-1)
        return merged[:, None, None, :]

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
        self.last_attention_mask = attention_mask
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

            if attention_mask is not None:
                attention_mask = self._merge_attention_mask(attention_mask, merge_fn)
                self.last_attention_mask = attention_mask
            
            t_merge_end = time.perf_counter()
            merge_time_s = t_merge_end - t_merge_start
            TOME_MERGE_TIME["total_s"] += merge_time_s
            TOME_MERGE_TIME["call_count"] += 1
        # ────────────────────────────────────────────────────────────────

        scale = math.sqrt(self.attention_head_size)
        scores = torch.matmul(q, k.transpose(-1, -2)) / scale

        if attention_mask is not None and attention_mask.shape[-1] != scores.shape[-1]:
            raise RuntimeError(
                "ToMe attention mask length does not match merged sequence length: "
                f"mask={attention_mask.shape[-1]}, scores={scores.shape[-1]}. "
                "The reduced mask must be propagated between encoder layers."
            )

        if attention_mask is not None:
            scores = scores + attention_mask

        probs = nn.functional.softmax(scores, dim=-1)
        probs  = self.dropout(probs)
        if head_mask is not None:
            probs = probs * head_mask
        ctx    = torch.matmul(probs, v)                 # (B, H, T', d)

        ctx = ctx.permute(0, 2, 1, 3).contiguous()
        ctx = ctx.view(ctx.size(0), ctx.size(1), self.all_head_size)

        # BertSelfOutput: dense -> dropout -> LayerNorm(x + residual)
        ctx = self.out_dense(ctx)
        ctx = self.out_dropout(ctx)
        attention_output = self.out_LayerNorm(ctx + residual)  # both T'

        outputs = (attention_output,)
        if output_attentions:
            outputs = outputs + (probs,)
        return outputs


class ToMeBertEncoder(nn.Module):
    """BertEncoder wrapper that carries ToMe's reduced mask layer by layer."""

    def __init__(self, original_encoder):
        super().__init__()
        self.config = original_encoder.config
        self.layer = original_encoder.layer
        self.gradient_checkpointing = getattr(original_encoder, "gradient_checkpointing", False)
        if hasattr(original_encoder, "_gradient_checkpointing_func"):
            self._gradient_checkpointing_func = original_encoder._gradient_checkpointing_func

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
        cache_position=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and getattr(self.config, "add_cross_attention", False) else None
        )
        next_decoder_cache = () if use_cache else None
        current_attention_mask = attention_mask

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                checkpoint_fn = getattr(self, "_gradient_checkpointing_func", None)
                if checkpoint_fn is not None:
                    layer_outputs = checkpoint_fn(
                        layer_module.__call__,
                        hidden_states,
                        current_attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states,
                        current_attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                    )
            else:
                try:
                    layer_outputs = layer_module(
                        hidden_states,
                        current_attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                        cache_position=cache_position,
                    )
                except TypeError:
                    layer_outputs = layer_module(
                        hidden_states,
                        current_attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                    )

            hidden_states = layer_outputs[0]
            current_attention_mask = getattr(
                layer_module.attention,
                "last_attention_mask",
                current_attention_mask,
            )

            if use_cache:
                next_decoder_cache = next_decoder_cache + (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if getattr(self.config, "add_cross_attention", False):
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                value
                for value in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if value is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


def patch_bert_with_tome(model: BertModel, r: int = 8) -> BertModel:
    """
    Replace every BertAttention block with ToMeBertAttention.
    Patching the full block (not just self-attn) is required so that
    the residual connection is also merged to match the reduced T'.
    """
    for layer in model.encoder.layer:
        layer.attention = ToMeBertAttention(layer.attention, r=r)
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
    B, T, _ = metric.shape

    with torch.no_grad():
        metric = metric / (metric.norm(dim=-1, keepdim=True) + 1e-6)

        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        node_max, node_idx = scores.max(dim=-1)

        node_max = node_max.clone()
        node_max[:, 0] = -float("inf")

        a_size = a.size(1)
        r = min(r, a_size - 1)

        sorted_idx = node_max.argsort(dim=-1, descending=True)
        src_idx = sorted_idx[..., :r]
        unm_idx = sorted_idx[..., r:]
        dst_idx = node_idx.gather(dim=-1, index=src_idx)

    def merge(x: Tensor, mode: str = "mean") -> Tensor:
        """Merge r token pairs; returns a tensor with T - r tokens."""
        src, dst = x[..., ::2, :], x[..., 1::2, :].clone()
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
        elif mode == "max":
            dst.scatter_reduce_(
                -2,
                dst_idx.unsqueeze(-1).expand(n, r, c),
                matched_src,
                reduce="amax",
                include_self=True,
            )
        else:
            raise ValueError(f"Unsupported merge mode: {mode}")
        unmerged = src.gather(
            dim=-2,
            index=unm_idx.unsqueeze(-1).expand(n, n_unm, c),
        )
        pos_a = unm_idx * 2
        pos_b = (torch.arange(dst.size(1), device=x.device)
                 .unsqueeze(0)
                 .expand(n, -1) * 2 + 1)
        out_tokens = torch.cat([unmerged, dst], dim=1)
        out_pos = torch.cat([pos_a, pos_b], dim=1)
        order = out_pos.argsort(dim=-1)
        return out_tokens.gather(
            dim=1,
            index=order.unsqueeze(-1).expand(n, out_tokens.size(1), c),
        )

    def unmerge(x: Tensor) -> Tensor:
        return x

    return merge, unmerge
