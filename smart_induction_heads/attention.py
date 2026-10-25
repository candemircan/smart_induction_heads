"""Chunk-level attention and the accuracies derived from it (paper App. A.2)."""

import torch
from einops import rearrange


def transitions(chunk_len: int, n_primitives: int) -> torch.Tensor:
    """Positions inside a 3rd-order chunk whose next token starts another 2nd-order chunk of the same 3rd-order chunk.

    Returns:
        torch.Tensor: position indices (n_primitives - 1,)
    """
    return torch.arange(chunk_len // n_primitives - 1, chunk_len - 1, chunk_len // n_primitives)


def pool_attention(attn: torch.Tensor, order: int, chunk_len: int, n_primitives: int) -> torch.Tensor:
    """Mean attention from the tokens of each chunk to the tokens of each chunk (Eq. 2).

    For 3rd-order sequences only the rows at `transitions` count, because only there does the
    3rd-order chunk decide the next token.

    Args:
        attn (torch.Tensor): attention weights (..., T, T)

    Returns:
        torch.Tensor: chunk attention (..., C, C) with C = T / chunk_len
    """
    attn = rearrange(attn, '... (i x) (j y) -> ... i j x y', x=chunk_len, y=chunk_len)
    if order == 3: attn = attn[..., transitions(chunk_len, n_primitives), :]
    return rearrange(attn, '... i j x y -> ... i j (x y)').mean(-1)


def context_accuracy(pooled: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Whether the most attended earlier chunk has the same chunk id as the current chunk.

    Args:
        pooled (torch.Tensor): chunk attention (..., C, C)
        ids (torch.Tensor): chunk ids, broadcastable to (..., C)

    Returns:
        torch.Tensor: accuracy per chunk, 0 for the first chunk (..., C)
    """
    C = pooled.size(-1)
    past = torch.ones(C, C, dtype=torch.bool).tril(-1)
    target = pooled.masked_fill(~past, -torch.inf)[..., 1:, :].argmax(-1)
    ids = ids.expand(*target.shape[:-1], C)
    acc = (ids[..., 1:] == ids.gather(-1, target)).float()
    return torch.cat([torch.zeros_like(acc[..., :1]), acc], -1)


def successor_accuracy(attn: torch.Tensor, seqs: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Whether each token attends most to a token whose predecessor equals it.

    Args:
        attn (torch.Tensor): attention weights (B, T, T)
        seqs (torch.Tensor): token ids (B, T)

    Returns:
        torch.Tensor: accuracy per token, 0 for the first chunk (B, T)
    """
    target = attn.argmax(-1)
    # target 0 wraps to the last token, as in the paper's analysis
    acc = (seqs.gather(1, (target - 1) % seqs.size(1)) == seqs).float()
    acc[:, :chunk_size] = 0
    return acc
