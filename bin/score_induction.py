"""Score every head on a repeated random token sequence against an ideal induction pattern (App. A.3).

Saves the scores (L, H) to data/induction_scores/<model>.pt.
"""

from argparse import ArgumentParser
from pathlib import Path

import torch
from einops import repeat
from transformers import AutoModelForCausalLM

from smart_induction_heads.analysis import model_id

parser = ArgumentParser()
parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B')
parser.add_argument('--length', type=int, default=64, help='length of the sequence that is repeated')
args = parser.parse_args()

torch.manual_seed(1234)
model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', torch_dtype=torch.bfloat16, attn_implementation='eager')

tokens = repeat(torch.randint(1000, 2000, (args.length,)), 't -> (2 t)')
seq = torch.cat([torch.tensor([model.config.bos_token_id]), tokens])[None].to(model.device)
with torch.no_grad(): attn = torch.stack(model(seq, output_attentions=True).attentions)[:, 0, :, 1:, 1:].double().cpu()  # (L, H, T, T) without <bos>

# each token of the second repetition should attend to the successor of its first occurrence
T = 2 * args.length
ideal = torch.zeros(T, T, dtype=torch.double)
query = torch.arange(args.length, T)
ideal[query, query - args.length + 1] = 1

scores = (attn * ideal).sum((-2, -1)) / attn.sum((-2, -1))
Path('data/induction_scores').mkdir(parents=True, exist_ok=True)
torch.save(scores.float(), f'data/induction_scores/{model_id(args.model)}.pt')
