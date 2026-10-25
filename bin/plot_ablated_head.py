"""Attention of one head with and without another head ablated, and its accuracy over chunks. Fig. 6."""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D
from nnsight import LanguageModel

from smart_induction_heads.attention import context_accuracy, pool_attention, successor_accuracy
from smart_induction_heads.heads import zero_heads
from smart_induction_heads.plotting import ATTENTION_CMAP, COLUMN_WIDTH, GREY, INK, RED, save
from smart_induction_heads.sequences import sample

parser = ArgumentParser()
parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B')
parser.add_argument('--head', default='14-3', help='layer-head to inspect')
parser.add_argument('--ablate', default='13-4', help='layer-head to ablate')
parser.add_argument('--n_seqs', type=int, default=84)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

task = dict(order=2, chunk_size=3, n_chunks=3, n_primitives=1, n_reps=4)
model = LanguageModel(args.model, device_map='auto', torch_dtype=torch.bfloat16, attn_implementation='eager', dispatch=True)
layer, head = map(int, args.head.split('-'))
ablate_layer, ablate_head = map(int, args.ablate.split('-'))
colors = {'context': RED, 'successor': INK}

fig, ax = plt.subplot_mosaic([[0, 1], [2, 2]], figsize=(COLUMN_WIDTH, 3.4), layout='constrained', height_ratios=[1.15, 1])
for col, (heads, title, style) in enumerate([({}, 'No ablation', '-'), ({ablate_layer: [ablate_head]}, f'Head {args.ablate} ablated', '--')]):
    torch.manual_seed(args.seed)
    attn, seqs, ids = [], [], []
    for _ in range(args.n_seqs // args.batch_size):
        s, i = sample(args.batch_size, torch.arange(model.config.vocab_size), **task)
        with torch.no_grad(), model.trace(s.to(model.device), output_attentions=True):
            zero_heads(model, heads)
            output = model.output.save()
        attn.append(output['attentions'][layer][:, head].float().cpu())
        seqs.append(s)
        ids.append(i)
    attn, seqs, ids = torch.cat(attn), torch.cat(seqs), torch.cat(ids)

    context = context_accuracy(pool_attention(attn, 2, task['chunk_size'], 1), ids)
    successor = successor_accuracy(attn, seqs, task['chunk_size'])
    ax[2].plot(torch.arange(context.size(1)), context.mean(0) * 100, color=colors['context'], linestyle=style)
    ax[2].plot(torch.arange(successor.size(1)) / task['chunk_size'], successor.mean(0) * 100, color=colors['successor'], linestyle=style)

    ax[col].imshow(attn[0], cmap=ATTENTION_CMAP)
    ax[col].set_title(f'{title}\n(head {args.head})', fontweight='normal')
    ax[col].set_xticks([])
    ax[col].set_yticks([])
    for spine in ax[col].spines.values(): spine.set_visible(True)
    for edge in range(task['chunk_size'], seqs.size(1), task['chunk_size']):
        ax[col].axvline(edge - 0.5, color='#d9d4ce', linewidth=0.3)
        ax[col].axhline(edge - 0.5, color='#d9d4ce', linewidth=0.3)

ax[2].axhline(100 / task['n_chunks'], color=GREY, lw=0.6, ls=':')
ax[2].set_ylabel('Attention\naccuracy (%)')
ax[2].set_ylim([0, 101])
ax[2].set_xlabel('Chunks seen')
ax[2].legend(handles=[Line2D([0], [0], color=colors['successor'], label='any successor'),
                      Line2D([0], [0], color=colors['context'], label='correct context'),
                      Line2D([0], [0], color=GREY, linestyle='-', label='no ablation'),
                      Line2D([0], [0], color=GREY, linestyle='--', label=f'{args.ablate} ablated')],
             loc='lower right', ncol=2, handlelength=1.75, handletextpad=0.4, columnspacing=1)
save(fig, 'ablated_head')
