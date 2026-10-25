"""Toy 1st-, 2nd- and 3rd-order sequences, and the attention of two heads to successors in the same and in another chunk type. Fig. 1.

The bars use new 2nd-order sequences with the settings of the learning experiment (Fig. 2A) but another seed, so that
they are not the sequences on which the heads were selected. For each query token, a key is a
successor if the key's previous token is an earlier instance of the query token. A successor is in the same chunk type
if that earlier instance lies in a chunk of the same type as the query. The value per sequence is the attention summed
over successors, averaged over the queries that have successors of both kinds, excluding the last token of each chunk.
"""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from transformers import AutoModelForCausalLM

from smart_induction_heads.plotting import BLUE, GREY, INK, RED, TEXT_WIDTH, save
from smart_induction_heads.sequences import TASKS, sample

parser = ArgumentParser()
parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B')
parser.add_argument('--heads', nargs=2, default=['19-3', '2-3'], help='layer-head')
parser.add_argument('--titles', nargs=2, default=['Adaptive\nhead 19-3', 'Static\nhead 2-3'])
parser.add_argument('--n_seqs', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', torch_dtype=torch.bfloat16, attn_implementation='eager')
heads = [tuple(map(int, h.split('-'))) for h in args.heads]
task = TASKS[2]
torch.manual_seed(args.seed)
seqs, ids = sample(args.n_seqs, torch.arange(model.config.vocab_size), **task)
types = ids.repeat_interleave(task['chunk_size'], dim=1)  # (B, T)
T = seqs.size(1)

query, previous = torch.arange(T)[:, None], torch.arange(T - 1)[None, :]
successor = (seqs[:, :, None] == seqs[:, None, :-1]) & (previous + 1 < query)  # (B, T, T - 1), key = previous + 1
same = types[:, :, None] == types[:, None, :-1]
masks = torch.stack([successor & same, successor & ~same])  # (2, B, T, T - 1)
valid = (query[:, 0] % task['chunk_size'] != task['chunk_size'] - 1) & masks.any(-1).all(0)  # (B, T)

attn = []
for batch in seqs.split(args.batch_size):
    with torch.no_grad(): attentions = model(batch.to(model.device), output_attentions=True).attentions
    attn.append(torch.stack([attentions[l][:, h] for l, h in heads]).float().cpu())
attn = torch.cat(attn, 1)[..., 1:]  # (heads, B, T, T - 1)
summed = torch.einsum('nbqk,cbqk->ncbq', attn, masks.float())  # (heads, kind, B, T)
per_seq = (summed * valid).sum(-1) / valid.sum(-1)  # (heads, kind, B)
mean, sem = per_seq.mean(-1), per_seq.std(-1) / args.n_seqs ** 0.5
for (l, h), m in zip(heads, mean): print(f'head {l}-{h}: same chunk type {m[0]:.3f}, other chunk type {m[1]:.3f}')

fig = plt.figure(figsize=(TEXT_WIDTH, 1.95))
ax_seq, ax_bar = fig.subplots(1, 2, gridspec_kw=dict(width_ratios=[2.6, 1], wspace=0.25))

# toy sequences, chunks as in Section 3: alpha=<a,b,c>, gamma=<c,b,a>; psi=<beta,alpha,gamma>, Omega=<gamma,alpha,beta>
rows = [
    ('1st order', [], [['a', 'b', 'c'], ['a', 'b', 'c'], ['a', 'b', '?']], {(0, 2): RED, (1, 2): RED}, 'a bi-gram model\nsuffices'),
    ('2nd order', [r'\alpha', r'\gamma', r'\alpha'], [['a', 'b', 'c'], ['c', 'b', 'a'], ['a', 'b', '?']], {(0, 2): RED, (1, 2): BLUE},
     'the successor depends\non the current context'),
    ('3rd order', [r'\Omega', r'\psi', r'\Omega'], [[r'\gamma', r'\alpha', r'\beta'], [r'\beta', r'\alpha', r'\gamma'], [r'\gamma', r'\alpha', '?']],
     {(0, 2): RED, (1, 2): BLUE}, '2nd-order chunks ($\\alpha$, $\\beta$, $\\gamma$)\nform higher-order chunks'),
]
box, gap, group_gap, row_gap, x0 = 1.0, 0.18, 0.75, 3.3, 4.1
group_width = 3 * box + 2 * gap + group_gap
for r, (name, labels, groups, colored, rule) in enumerate(rows):
    y = -r * row_gap
    ax_seq.text(0, y + box / 2, name, ha='left', va='center', fontsize=7.5, fontweight='bold')
    centers = {}
    for g, items in enumerate(groups):
        gx = x0 + g * group_width
        if labels: ax_seq.text(gx + 1.5 * box + gap, y + box + 0.45, f'${labels[g]}$', ha='center', va='center', fontsize=9, color=GREY)
        for i, item in enumerate(items):
            x = gx + i * (box + gap)
            centers[(g, i)] = x + box / 2
            face = colored.get((g, i), '#ebe6df')
            current = (g, i) == (2, 1)
            ax_seq.add_patch(FancyBboxPatch((x, y), box, box, boxstyle='round,pad=0,rounding_size=0.18', fc='white' if current or item == '?' else face,
                                            ec=INK if current else 'none', lw=1.1 if current else 0))
            ax_seq.text(x + box / 2, y + box / 2 - 0.04, item if item == '?' else f'${item}$', ha='center', va='center', fontsize=10,
                        color='white' if face in (RED, BLUE) else INK)
    ax_seq.text(x0 + 3 * group_width - group_gap + 0.7, y + box / 2, rule, ha='left', va='center', fontsize=6.5, linespacing=1.3)
    for (g, i), color in colored.items():
        ax_seq.add_patch(FancyArrowPatch((centers[(2, 1)], y - 0.12), (centers[(g, i)], y - 0.12), connectionstyle='arc3,rad=-0.22',
                                         arrowstyle='-|>,head_length=0.6,head_width=0.35', mutation_scale=6.9, lw=0.9, color=color))
ax_seq.legend(handles=[Patch(fc='white', ec=INK, lw=0.9), Patch(fc=RED), Patch(fc=BLUE)],
              labels=['current token', 'successor in correct context', 'successor in wrong context'], loc='lower center',
              bbox_to_anchor=(0.5, -0.06), ncol=3, handlelength=1.0, handleheight=1.0, handletextpad=0.5, columnspacing=1.2)
ax_seq.set_xlim(-0.1, x0 + 3 * group_width - group_gap + 8.3)
ax_seq.set_ylim(-2 * row_gap - 2.9, box + 1.0)
ax_seq.set_aspect('equal')
ax_seq.axis('off')

for k, (color, label) in enumerate([(RED, 'correct successor'), (BLUE, 'wrong successor')]):
    ax_bar.bar(torch.arange(2) + (k - 0.5) * 0.38, mean[:, k], width=0.36, yerr=sem[:, k], color=color, label=label, ec=INK, lw=0.5,
               error_kw=dict(lw=0.8, ecolor=INK))
ax_bar.set_xticks(range(2), args.titles)
ax_bar.tick_params(axis='x', length=0)
ax_bar.set_ylabel('Attention')
ax_bar.set_ylim(0, 1)
ax_bar.set_yticks([0, 0.5, 1])
ax_bar.legend(loc='upper right', handlelength=1, bbox_to_anchor=(1.1, 1.12))
save(fig, 'overview')
