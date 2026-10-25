"""Head selection for ablations (paper Sec. 4.3, 5.2 and App. A.5) and the ablation itself."""

from collections import defaultdict

import torch

INDUCTION_THRESHOLD = 0.4  # induction score (App. A.3)
CONTEXT_THRESHOLD = 0.85  # 2nd-order context decodability (Sec. 5.2)
CONTEXT_POOL_THRESHOLD = 0.55  # random control heads must decode context below this


def heads_above(scores: torch.Tensor, threshold: float) -> dict[int, list[int]]:
    """Heads whose score exceeds `threshold`.

    Args:
        scores (torch.Tensor): score per head (L, H)

    Returns:
        dict[int, list[int]]: heads per layer
    """
    heads = defaultdict(list)
    for layer, head in (scores > threshold).nonzero().tolist(): heads[layer].append(head)
    return dict(heads)


def random_heads(scores: torch.Tensor, threshold: float, pool_threshold: float) -> dict[int, list[int]]:
    """As many random heads with score <= `pool_threshold` as there are heads with score > `threshold`.

    Args:
        scores (torch.Tensor): score per head (L, H)

    Returns:
        dict[int, list[int]]: heads per layer
    """
    n = int((scores > threshold).sum())
    perm = torch.randperm(scores.numel())
    # the paper's runs index the permutation with itself; this keeps their draws
    eligible = [i for i in perm[perm].tolist() if scores.flatten()[i] <= pool_threshold]
    heads = defaultdict(list)
    for i in eligible[:n]: heads[i // scores.size(1)].append(i % scores.size(1))
    return dict(heads)


def layer_matched_random_heads(scores: torch.Tensor, reference: dict[int, list[int]], pool_threshold: float) -> dict[int, list[int]]:
    """For each layer of `reference`, as many random heads with score <= `pool_threshold` from the same layer (App. A.5).

    A shortfall is filled from the next, then the previous layer. If that is not enough, the layer is
    redrawn with the threshold raised by 0.05, up to 0.2. Any heads still missing come from all layers.

    Args:
        scores (torch.Tensor): score per head (L, H)
        reference (dict[int, list[int]]): heads per layer to match

    Returns:
        dict[int, list[int]]: heads per layer
    """
    L, H = scores.shape
    used = []

    def draw(candidates, n):
        picked = [candidates[i] for i in torch.randperm(len(candidates))[:n].tolist()]
        used.extend(picked)
        return len(picked)

    for layer, ref in reference.items():
        n = len(ref)
        for increment in [0.0, 0.05, 0.1, 0.15, 0.2]:
            attempt_start, found = len(used), 0
            threshold = pool_threshold + increment
            own = [(layer, h) for h in range(H) if scores[layer, h] <= threshold and (layer, h) not in used]
            if len(own) >= n:
                found = draw(own, n)
                break
            used.extend(own)
            found = len(own)
            for adjacent in [layer + 1, layer - 1]:
                if not 0 <= adjacent < L or found == n: continue
                candidates = [(adjacent, h) for h in range(H) if scores[adjacent, h] <= threshold and (adjacent, h) not in used]
                if candidates: found += draw(candidates, n - found)
            if found == n: break
            if increment < 0.2: del used[attempt_start:]
        if found < n:
            candidates = [(l, h) for l in range(L) for h in range(H) if scores[l, h] <= pool_threshold and (l, h) not in used]
            if candidates: draw(candidates, n - found)

    heads = defaultdict(list)
    for layer, head in used: heads[layer].append(head)
    return dict(heads)


def zero_heads(model, heads: dict[int, list[int]]):
    """Zero the outputs of `heads` before each layer's output projection. Call inside `model.trace` (nnsight)."""
    n_heads = model.config.num_attention_heads
    for layer, hs in heads.items():
        if not hs: continue
        o_proj = model.model.layers[layer].self_attn.o_proj
        x = o_proj.input
        x = x.view(x.size(0), x.size(1), n_heads, x.size(-1) // n_heads)
        x[:, :, hs] *= 0
        o_proj.input = x.view(x.size(0), x.size(1), -1)
