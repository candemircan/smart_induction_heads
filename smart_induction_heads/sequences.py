"""Hierarchical token sequences (paper Sec. 3 and App. A.1)."""

import torch
from einops import rearrange

# Table 1 of the paper. `chunk_size` is V, `n_chunks` is P (2nd order) or P' (3rd order),
# `n_primitives` is P (3rd order only), `n_reps` is N.
TASKS = {
    2: dict(order=2, chunk_size=8, n_chunks=4, n_primitives=1, n_reps=8),
    3: dict(order=3, chunk_size=4, n_chunks=4, n_primitives=4, n_reps=8),
}


def unique_permutations(n: int, k: int) -> torch.Tensor:
    """Draw `k` distinct random permutations of `range(n)`.

    Returns:
        torch.Tensor: permutations (k, n)
    """
    perms = []
    while len(perms) < k:
        perm = torch.randperm(n)
        if not any(torch.equal(perm, p) for p in perms): perms.append(perm)
    return torch.stack(perms)


def chunk_order(n_chunks: int, n_reps: int) -> torch.Tensor:
    """Shuffle `n_reps` repetitions of each of `n_chunks` chunk ids.

    Returns:
        torch.Tensor: chunk id at each chunk position (n_chunks * n_reps,)
    """
    return (torch.arange(n_reps * n_chunks) % n_chunks)[torch.randperm(n_reps * n_chunks)]


def second_order(tokens: torch.Tensor, n_chunks: int, n_reps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate shuffled repetitions of `n_chunks` distinct permutations of `tokens`.

    Args:
        tokens (torch.Tensor): token ids (V,)

    Returns:
        seq (torch.Tensor): token sequence (n_chunks * n_reps * V,)
        ids (torch.Tensor): chunk id at each chunk position (n_chunks * n_reps,)
    """
    chunks = tokens[unique_permutations(len(tokens), n_chunks)]
    ids = chunk_order(n_chunks, n_reps)
    return chunks[ids].flatten(), ids


def third_order(tokens: torch.Tensor, n_chunks: int, n_primitives: int, n_reps: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate shuffled repetitions of `n_chunks` distinct orderings of `n_primitives` permutations of `tokens`.

    Args:
        tokens (torch.Tensor): token ids (V,)

    Returns:
        seq (torch.Tensor): token sequence (n_chunks * n_reps * n_primitives * V,)
        ids (torch.Tensor): 3rd-order chunk id at each chunk position (n_chunks * n_reps,)
        primitive_ids (torch.Tensor): 2nd-order chunk ids inside each chunk position (n_chunks * n_reps, n_primitives)
    """
    primitives = tokens[unique_permutations(len(tokens), n_primitives)]
    compositions = unique_permutations(n_primitives, n_chunks)
    chunks = rearrange(primitives[compositions], 'c p v -> c (p v)')
    ids = chunk_order(n_chunks, n_reps)
    return chunks[ids].flatten(), ids, compositions[ids]


def add_noise(seq: torch.Tensor, pool: torch.Tensor, p: float) -> torch.Tensor:
    """Replace each token after the first, with probability `p`, by a different token from `pool`.

    Args:
        seq (torch.Tensor): token sequence (T,)
        pool (torch.Tensor): candidate replacement tokens (V,)
    """
    seq = seq.clone()
    # a loop keeps the random draws in the order that generated the paper's noisy sequences
    for i in range(1, len(seq)):
        if torch.rand(1).item() < p:
            others = pool[pool != seq[i]]
            seq[i] = others[torch.randint(len(others), (1,))].item()
    return seq


def sample(n: int, vocab: torch.Tensor, order: int, chunk_size: int, n_chunks: int, n_primitives: int, n_reps: int, noise: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `n` sequences, each from its own `chunk_size` tokens drawn from `vocab`.

    Args:
        vocab (torch.Tensor): token ids to draw from (n_vocab,)

    Returns:
        seqs (torch.Tensor): token sequences (n, T)
        ids (torch.Tensor): chunk id at each chunk position (n, n_chunks * n_reps)
    """
    seqs, ids = [], []
    for _ in range(n):
        tokens = vocab[torch.randint(len(vocab), (chunk_size,))]
        if order == 2: seq, chunk_ids = second_order(tokens, n_chunks, n_reps)
        else: seq, chunk_ids, _ = third_order(tokens, n_chunks, n_primitives, n_reps)
        if noise > 0: seq = add_noise(seq, tokens, noise)
        seqs.append(seq)
        ids.append(chunk_ids)
    return torch.stack(seqs), torch.stack(ids)
