"""Run a trained small transformer on 3rd-order sequences (Sec. 6).

Saves next-token accuracy and each head's context accuracy, as `bin/evaluate.py` does,
to data/runs/none/markov3/<run>/.
"""

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from einops import rearrange

from smart_induction_heads.attention import context_accuracy, pool_attention
from smart_induction_heads.model import Transformer
from smart_induction_heads.sequences import sample

parser = ArgumentParser()
parser.add_argument('--run', required=True, help='directory with args.json and weights.pt')
parser.add_argument('--n_seqs', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

run = Path(args.run)
config = Namespace(**json.loads((run / 'args.json').read_text()))
# initialising the model draws random numbers; seeding first keeps the paper's sequences
torch.manual_seed(args.seed)
model = Transformer(config)
model.load_state_dict(torch.load(run / 'weights.pt', map_location='cpu'))
model.eval()
task = dict(order=3, chunk_size=config.chunk_size, n_chunks=config.n_permute, n_primitives=config.n_permute_primitive, n_reps=config.n_reps)
chunk_len = config.chunk_size * config.n_permute_primitive

model_accs, head_accs = [], []
for _ in range(args.n_seqs // args.batch_size):
    seqs, ids = sample(args.batch_size, torch.arange(config.vocab_size), **task)
    with torch.no_grad(): logits, attn = model(seqs, return_attn=True)
    head_accs.append(context_accuracy(pool_attention(attn, 3, chunk_len, config.n_permute_primitive), rearrange(ids, 'b c -> 1 b 1 c')))
    model_accs.append((logits.argmax(-1)[:, :-1] == seqs[:, 1:]).float())

out = Path('data/runs/none/markov3') / run.name
out.mkdir(parents=True, exist_ok=True)
torch.save(torch.cat(model_accs), out / 'model_accs.pt')
torch.save(rearrange(torch.cat(head_accs, 1), 'l b h c -> l h b c'), out / 'head_accs.pt')
(out / 'args.json').write_text(json.dumps(vars(args), indent=2))
