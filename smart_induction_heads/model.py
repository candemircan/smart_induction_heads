"""Small decoder-only transformer with rotary position embeddings (paper Sec. 6).

Adapted from nanoGPT (https://github.com/karpathy/nanoGPT/blob/master/model.py) and the rotary
embeddings of RoFormer (https://arxiv.org/abs/2104.09864, Eq. 34). Parameter names match the trained checkpoints.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def rotary_frequencies(dim: int, positions: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """Complex rotations per position and frequency (T, dim / 2)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    return torch.polar(torch.ones(len(positions), dim // 2), torch.outer(positions, freqs))


def rotate(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to queries or keys `x` (B, T, nh, hs) with `freqs` (T, hs / 2)."""
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(x_ * freqs[None, :, None]).flatten(3).type_as(x)


def init_weights(module: nn.Module):
    if isinstance(module, (nn.Linear, nn.Embedding)): nn.init.normal_(module.weight, mean=0.0, std=0.02)
    if isinstance(module, nn.Linear) and module.bias is not None: nn.init.zeros_(module.bias)


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.Tensor, freqs: torch.Tensor, return_attn: bool = False):
        """x (torch.Tensor): input (B, T, n_embd). Returns the output and, if asked, the attention weights (B, nh, T, T)."""
        q, k, v = (rearrange(t, 'b t (h d) -> b t h d', h=self.n_head) for t in self.c_attn(x).split(self.n_embd, dim=2))
        q, k = rotate(q, freqs), rotate(k, freqs)
        q, k, v = (rearrange(t, 'b t h d -> b h t d') for t in (q, k, v))
        if return_attn:
            T = x.size(1)
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
            attn = attn.masked_fill(~torch.ones(T, T, dtype=torch.bool, device=x.device).tril(), float('-inf')).softmax(-1)
            y = attn @ v
        else:
            attn, y = None, F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(rearrange(y, 'b h t d -> b t (h d)')), attn


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = nn.Sequential()
        self.mlp.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.mlp.gelu = nn.GELU()
        self.mlp.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x, freqs, return_attn=False):
        y, attn = self.attn(self.ln_1(x), freqs, return_attn)
        x = x + y
        return x + self.mlp(self.ln_2(x)), attn


class Transformer(nn.Module):
    """`config` needs `vocab_size`, `block_size`, `n_layer`, `n_head`, `n_embd` and `bias`."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.register_buffer('freqs_cis_t', rotary_frequencies(config.n_embd // config.n_head, torch.arange(config.block_size, dtype=torch.float32) * 2))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(init_weights)
        # scaled init of the residual projections (GPT-2)
        for name, p in self.named_parameters():
            if name.endswith('c_proj.weight'): nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def forward(self, idx: torch.Tensor, return_attn: bool = False):
        """idx (torch.Tensor): token ids (B, T). Returns logits (B, T, vocab) and, if asked, attention weights per layer (L, B, nh, T, T)."""
        freqs = self.freqs_cis_t[:idx.size(1)]
        x, attns = self.transformer.wte(idx), []
        for block in self.transformer.h:
            x, attn = block(x, freqs, return_attn)
            attns.append(attn)
        logits = self.lm_head(self.transformer.ln_f(x))
        return (logits, torch.stack(attns)) if return_attn else logits
