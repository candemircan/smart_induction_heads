"""Load evaluation runs and summarise them for the figures."""

from pathlib import Path

import torch
from einops import rearrange

from .attention import transitions
from .heads import INDUCTION_THRESHOLD
from .sequences import TASKS

LEARNING_THRESHOLD = 0.4  # mean context accuracy of a generic in-context learning head


def model_id(model: str) -> str: return model.split('/')[-1]


def load_run(name: str, order: int, model: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a run of `bin/evaluate.py`.

    Returns:
        model_accs (torch.Tensor): next-token accuracy (B, T - 1)
        head_accs (torch.Tensor): context accuracy per head and chunk, NaN for heads not stored (L, H, B, C)
    """
    path = Path('data/runs') / name / f'markov{order}' / model_id(model)
    return torch.load(path / 'model_accs.pt'), torch.load(path / 'head_accs.pt')


def induction_scores(model: str) -> torch.Tensor:
    """Induction score per head (L, H)."""
    return torch.load(f'data/induction_scores/{model_id(model)}.pt')


def lm_accuracy(model_accs: torch.Tensor, order: int, task: dict | None = None) -> torch.Tensor:
    """Mean next-token accuracy per chunk over the tokens that the chunk determines.

    These are all but the last token of a 2nd-order chunk, and the `transitions` of a 3rd-order chunk.

    Args:
        model_accs (torch.Tensor): next-token accuracy (B, T - 1)
        task (dict): sequence parameters as in `TASKS`, which supplies the default

    Returns:
        torch.Tensor: accuracy per chunk (B, C)
    """
    task = task or TASKS[order]
    chunk_len = task['chunk_size'] * task['n_primitives']
    accs = torch.cat([model_accs, torch.ones(model_accs.size(0), 1)], -1)
    accs = rearrange(accs, 'b (c x) -> b c x', x=chunk_len)
    positions = torch.arange(chunk_len - 1) if order == 2 else transitions(chunk_len, task['n_primitives'])
    return accs[..., positions].mean(-1)


def mean_sem(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and standard error over the first axis, ignoring rows with NaN.

    Args:
        x (torch.Tensor): values (n, ...)
    """
    x = x[~x.flatten(1).isnan().any(1)]
    return x.mean(0), x.std(0) / len(x) ** 0.5


def candidates(head_accs: torch.Tensor, induction: torch.Tensor, kind: str) -> torch.Tensor:
    """Induction heads, or generic heads: non-induction heads with mean context accuracy above `LEARNING_THRESHOLD`.

    Args:
        head_accs (torch.Tensor): context accuracy (L, H, B, C)
        induction (torch.Tensor): induction score (L, H)

    Returns:
        torch.Tensor: mask (L, H)
    """
    is_induction = induction > INDUCTION_THRESHOLD
    if kind == 'induction': return is_induction
    return ~is_induction & (head_accs.mean(-1).mean(-1) > LEARNING_THRESHOLD)


def loo_top_heads(head_accs: torch.Tensor, mask: torch.Tensor, kind: str, k: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """Leave one sequence out: rank the heads in `mask` on the other sequences and average the top `k` on the left-out one (App. A.2).

    Generic heads must also exceed `LEARNING_THRESHOLD` on the other sequences.

    Args:
        head_accs (torch.Tensor): context accuracy (L, H, B, C)
        mask (torch.Tensor): candidate heads (L, H)

    Returns:
        curves (torch.Tensor): mean context accuracy of the selected heads on each left-out sequence, NaN if none qualify (B, C)
        selected (torch.Tensor): flat indices of the selected heads per left-out sequence, -1 if fewer than `k` (B, k)
    """
    accs = rearrange(head_accs, 'l h b c -> (l h) b c')
    per_seq = accs.mean(-1)
    B = per_seq.size(1)
    others = torch.arange(B).expand(B, B)[~torch.eye(B, dtype=torch.bool)].view(B, B - 1)
    train = per_seq[:, others].mean(-1)  # (heads, B)
    eligible = mask.flatten()[:, None].expand_as(train)
    if kind == 'generic': eligible = eligible & (train > LEARNING_THRESHOLD)
    order = train.masked_fill(~eligible, -torch.inf).sort(dim=0, descending=True, stable=True).indices[:k]
    valid = eligible.gather(0, order)
    left_out = torch.where(valid[..., None], accs[order, torch.arange(B)], 0)  # (k, B, C)
    curves = left_out.sum(0) / valid.sum(0)[:, None]
    return curves, torch.where(valid, order, -1).T


def top_heads(head_accs: torch.Tensor, mask: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Context accuracy of the `k` heads in `mask` with the highest mean context accuracy.

    Args:
        head_accs (torch.Tensor): context accuracy (L, H, B, C)
        mask (torch.Tensor): candidate heads (L, H)

    Returns:
        torch.Tensor: context accuracy of the selected heads (k, B, C)
    """
    accs = rearrange(head_accs, 'l h b c -> (l h) b c')
    score = accs.mean((1, 2)).masked_fill(~mask.flatten(), -torch.inf)
    return accs[score.topk(k).indices]
