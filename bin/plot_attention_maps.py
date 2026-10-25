"""Attention maps of two heads on one short sequence: chunk level (top) and token level (bottom). Fig. 2B and Fig. 4."""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
from einops import reduce
from transformers import AutoModelForCausalLM

from smart_induction_heads.plotting import ATTENTION_CMAP, TEXT_WIDTH, save
from smart_induction_heads.sequences import second_order, third_order

parser = ArgumentParser()
parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B')
parser.add_argument('--order', type=int, default=2, choices=[2, 3])
parser.add_argument('--heads', nargs=2, default=['19-3', '2-3'], help='layer-head')
parser.add_argument('--titles', nargs=2, default=['Adaptive induction head', 'Static induction head'])
parser.add_argument('--out', default='induction_heads', help='file name in figures/, without suffix')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

chunk_size, n_chunks, n_primitives, n_reps = 3, 3, 3, 4
torch.manual_seed(args.seed)
model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', torch_dtype=torch.bfloat16, attn_implementation='eager')
tokens = torch.randint(model.config.vocab_size, (chunk_size,))
if args.order == 2:
    seq, ids = second_order(tokens, n_chunks, n_reps)
    chunk_len, letters = chunk_size, [chr(945 + i) for i in range(n_chunks)]
else:
    seq, ids, _ = third_order(tokens, n_chunks, n_primitives, n_reps)
    chunk_len, letters = chunk_size * n_primitives, ['Ω', 'ψ', 'φ']
labels = [f'${letters[i]}$' for i in ids]
with torch.no_grad(): attentions = model(seq[None].to(model.device), output_attentions=True).attentions

fig, ax = plt.subplots(2, 2, figsize=(0.49 * TEXT_WIDTH, 3.1), layout='constrained')
for col, (address, title) in enumerate(zip(args.heads, args.titles)):
    layer, head = map(int, address.split('-'))
    attn = attentions[layer][0, head].float().cpu()
    ax[0, col].imshow(reduce(attn, '(i x) (j y) -> i j', 'mean', x=chunk_len, y=chunk_len), cmap=ATTENTION_CMAP)
    ax[0, col].set_title(f'{title}\n(head {layer}-{head})')
    ax[0, col].set_xticks(range(len(labels)), labels)
    ax[0, col].set_yticks(range(len(labels)), labels)
    ax[1, col].imshow(attn, cmap=ATTENTION_CMAP)
    # one tick per chunk, near its centre
    ticks = torch.arange(0, len(seq), chunk_len) + (1 if args.order == 2 else chunk_size + 1)
    ax[1, col].set_xticks(ticks, labels)
    ax[1, col].set_yticks(ticks, labels)
    for edge in range(chunk_len, len(seq), chunk_len):
        ax[1, col].axvline(edge - 0.5, color='#d9d4ce', linewidth=0.3)
        ax[1, col].axhline(edge - 0.5, color='#d9d4ce', linewidth=0.3)
for a in ax.flat:
    a.tick_params(length=0, labelsize=5.5)
    for spine in a.spines.values(): spine.set_visible(True)
ax[0, 0].set_ylabel('Chunks')
ax[1, 0].set_ylabel('Tokens')
save(fig, args.out)
