"""Train a small transformer from scratch on new 3rd-order sequences at every step (Sec. 6).

Saves args.json, weights.pt and log.json (loss every 100 steps) to data/transformers/<run>/.
"""

import json
import math
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn.functional as F

from smart_induction_heads.model import Transformer
from smart_induction_heads.sequences import sample

parser = ArgumentParser()
parser.add_argument('--n_layer', type=int, default=2)
parser.add_argument('--n_embd', type=int, default=1280)
parser.add_argument('--n_head', type=int, default=16)
parser.add_argument('--chunk_size', type=int, default=8)
parser.add_argument('--n_permute', type=int, default=8, help='number of 3rd-order chunks')
parser.add_argument('--n_permute_primitive', type=int, default=8, help='number of 2nd-order chunks')
parser.add_argument('--n_reps', type=int, default=8)
parser.add_argument('--vocab_size', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
parser.add_argument('--iters', type=int, default=60000)
parser.add_argument('--learning_rate', type=float, default=6e-4)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
args.bias = False
args.block_size = args.chunk_size * args.n_permute_primitive * args.n_permute * args.n_reps
task = dict(order=3, chunk_size=args.chunk_size, n_chunks=args.n_permute, n_primitives=args.n_permute_primitive, n_reps=args.n_reps)

run = (f'markov=3_chunk_size={args.chunk_size}_npermprim={args.n_permute_primitive}_nperm={args.n_permute}_nreps={args.n_reps}'
       f'_nlayer={args.n_layer}_nembd={args.n_embd}_nhead={args.n_head}_blocksize={args.block_size}_seed={args.seed}')
out = Path('data/transformers') / run
out.mkdir(parents=True, exist_ok=True)
(out / 'args.json').write_text(json.dumps(vars(args), indent=2))

torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
model = Transformer(args).cuda()
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW([{'params': [p for p in params if p.dim() >= 2], 'weight_decay': args.weight_decay},
                               {'params': [p for p in params if p.dim() < 2], 'weight_decay': 0.0}],
                              lr=args.learning_rate, betas=(0.9, 0.95), fused=True)


def learning_rate(step: int) -> float:
    """Linear warm-up over the first 1% of steps, then cosine decay to 10% of the peak."""
    warmup, peak = int(args.iters * 0.01), args.learning_rate
    if step < warmup: return peak * step / warmup
    return 0.1 * peak + 0.5 * (1 + math.cos(math.pi * (step - warmup) / (args.iters - warmup))) * 0.9 * peak


losses = []
for step in range(args.iters):
    for group in optimizer.param_groups: group['lr'] = learning_rate(step)
    loss_sum = 0.0
    for _ in range(args.gradient_accumulation_steps):
        seqs, _ = sample(args.batch_size, torch.arange(args.vocab_size), **task)
        seqs = seqs.cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(seqs[:, :-1])
            loss = F.cross_entropy(logits.flatten(0, 1), seqs[:, 1:].flatten()) / args.gradient_accumulation_steps
        loss.backward()
        loss_sum += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    optimizer.zero_grad()
    if step % 100 == 0:
        losses.append(loss_sum)
        print(f'step {step} loss {loss_sum:.4f}', flush=True)

(out / 'log.json').write_text(json.dumps({'loss': losses}))
torch.save(model.state_dict(), out / 'weights.pt')
