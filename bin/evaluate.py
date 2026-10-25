"""Run a language model on hierarchical sequences, optionally with heads ablated.

Saves next-token accuracy and each head's context accuracy (App. A.2) to
data/runs/<condition>/markov<order>/<model>/. With --chunk_sizes, the chunk length varies between sequences, so
next-token accuracy is saved per chunk (lm_accs.pt) instead of per token (model_accs.pt).
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import torch
from einops import rearrange
from nnsight import LanguageModel

from smart_induction_heads.analysis import lm_accuracy, model_id
from smart_induction_heads.attention import context_accuracy, pool_attention
from smart_induction_heads.heads import (
    CONTEXT_POOL_THRESHOLD,
    CONTEXT_THRESHOLD,
    INDUCTION_THRESHOLD,
    heads_above,
    layer_matched_random_heads,
    random_heads,
    zero_heads,
)
from smart_induction_heads.sequences import TASKS, sample

parser = ArgumentParser()
parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B')
parser.add_argument('--order', type=int, default=2, choices=[2, 3])
parser.add_argument('--ablate', default='none', choices=['none', 'induction', 'context', 'random-induction', 'random-context'],
                    help='random-* ablates as many random heads as there are target heads, redrawn for every sequence')
parser.add_argument('--matched', action='store_true', help='draw random heads from the layers of the target heads (App. A.5)')
parser.add_argument('--noise', type=float, default=0.0, help='probability of replacing each token (App. B.2)')
parser.add_argument('--chunk_sizes', type=int, nargs=2, metavar=('LOW', 'HIGH'), help='draw each sequence\'s chunk size uniformly from LOW to HIGH (App. A.5)')
parser.add_argument('--n_seqs', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

name = args.ablate + '-matched' * args.matched + f'-noise{args.noise}' * (args.noise > 0) + (f'-chunks{args.chunk_sizes[0]}-{args.chunk_sizes[1]}' if args.chunk_sizes else '')
out = Path('data/runs') / name / f'markov{args.order}' / model_id(args.model)
out.mkdir(parents=True, exist_ok=True)

torch.manual_seed(args.seed)
model = LanguageModel(args.model, device_map='auto', torch_dtype=torch.bfloat16, attn_implementation='eager', dispatch=True)
task = TASKS[args.order]
vocab = torch.arange(model.config.vocab_size)

target = args.ablate.removeprefix('random-')
if target == 'induction': scores, threshold, pool = torch.load(f'data/induction_scores/{model_id(args.model)}.pt'), INDUCTION_THRESHOLD, INDUCTION_THRESHOLD
# context heads always come from the first train/test split of the 2nd-order decoding, for both orders
if target == 'context': scores, threshold, pool = torch.load(f'data/decoding/markov2/{model_id(args.model)}.pt')[0], CONTEXT_THRESHOLD, CONTEXT_POOL_THRESHOLD

random = args.ablate.startswith('random')
batch_size = 1 if random or args.chunk_sizes else args.batch_size
model_accs, head_accs = [], []
for _ in range(args.n_seqs // batch_size):
    if args.ablate == 'none': heads = {}
    elif not random: heads = heads_above(scores, threshold)
    elif args.matched: heads = layer_matched_random_heads(scores, heads_above(scores, threshold), pool)
    else: heads = random_heads(scores, threshold, pool)

    if args.chunk_sizes: task = TASKS[args.order] | dict(chunk_size=int(torch.randint(args.chunk_sizes[0], args.chunk_sizes[1] + 1, ())))
    chunk_len = task['chunk_size'] * task['n_primitives']
    seqs, ids = sample(batch_size, vocab, noise=args.noise, **task)
    seqs = seqs.to(model.device)
    with torch.no_grad(), model.trace(seqs, output_attentions=True):
        zero_heads(model, heads)
        output = model.output.save()

    pooled = torch.stack([pool_attention(a, args.order, chunk_len, task['n_primitives']) for a in output['attentions']]).float().cpu()
    head_accs.append(context_accuracy(pooled, rearrange(ids, 'b c -> 1 b 1 c')))
    accs = (output['logits'].argmax(-1)[:, :-1] == seqs[:, 1:]).float().cpu()
    model_accs.append(lm_accuracy(accs, args.order, task) if args.chunk_sizes else accs)

torch.save(torch.cat(model_accs), out / ('lm_accs.pt' if args.chunk_sizes else 'model_accs.pt'))
torch.save(rearrange(torch.cat(head_accs, 1), 'l b h c -> l h b c'), out / 'head_accs.pt')
(out / 'args.json').write_text(json.dumps(vars(args), indent=2))
