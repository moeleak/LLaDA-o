"""Small FlashAttention API fallback for environments without flash-attn.

The training environment on some Slurm images has PyTorch's fused SDPA kernels
but does not ship a flash-attn wheel.  LLaDA-o uses the packed varlen API for
both the language and SigLIP towers, so keep the same packed tensor contract
while dispatching each sequence to PyTorch SDPA.  This is an exact attention
fallback (not an approximation); installations with flash-attn continue to
use the original kernel.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.nn.functional import scaled_dot_product_attention


def _sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool,
    dropout_p: float,
    softmax_scale: Optional[float],
) -> torch.Tensor:
    """Apply SDPA to one packed sequence in ``(length, heads, dim)`` layout."""

    q_len = query.shape[0]
    kv_len = key.shape[0]
    query = query.transpose(0, 1).unsqueeze(0)
    key = key.transpose(0, 1).unsqueeze(0)
    value = value.transpose(0, 1).unsqueeze(0)

    if query.shape[1] != key.shape[1]:
        if query.shape[1] % key.shape[1] != 0:
            raise ValueError("query heads must be divisible by key/value heads")
        groups = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)

    # FlashAttention uses bottom-right causal alignment when q_len != kv_len.
    # SDPA's is_causal mask is top-left aligned for those shapes, so construct
    # the equivalent boolean mask explicitly in that case.
    if causal and q_len != kv_len:
        row = torch.arange(q_len, device=query.device)[:, None]
        col = torch.arange(kv_len, device=query.device)[None, :]
        attn_mask = col <= row + (kv_len - q_len)
        causal = False
    else:
        attn_mask = None

    kwargs = {
        "attn_mask": attn_mask,
        "dropout_p": dropout_p,
        "is_causal": causal,
    }
    if softmax_scale is not None:
        # ``scale`` is available in the PyTorch versions used by the Slurm
        # image.  Keep a compatibility fallback for older local test envs.
        try:
            output = scaled_dot_product_attention(**kwargs, scale=softmax_scale, query=query, key=key, value=value)
        except TypeError:
            output = scaled_dot_product_attention(
                query * (softmax_scale * query.shape[-1] ** 0.5),
                key,
                value,
                **kwargs,
            )
    else:
        output = scaled_dot_product_attention(query=query, key=key, value=value, **kwargs)
    return output.squeeze(0).transpose(0, 1)


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    **_: object,
) -> torch.Tensor:
    """Drop-in packed varlen FlashAttention interface backed by PyTorch SDPA."""

    del max_seqlen_q, max_seqlen_k
    q_offsets = cu_seqlens_q.to(device=q.device, dtype=torch.long)
    k_offsets = cu_seqlens_k.to(device=k.device, dtype=torch.long)
    if q_offsets.numel() != k_offsets.numel():
        raise ValueError("query and key cumulative sequence lengths must have equal batch size")

    outputs = []
    for index in range(q_offsets.numel() - 1):
        q_start, q_end = q_offsets[index].item(), q_offsets[index + 1].item()
        k_start, k_end = k_offsets[index].item(), k_offsets[index + 1].item()
        outputs.append(
            _sdpa(
                q[q_start:q_end],
                k[k_start:k_end],
                v[k_start:k_end],
                causal=causal,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
            )
        )
    return torch.cat(outputs, dim=0) if outputs else q.new_empty((0, q.shape[1], q.shape[2]))
