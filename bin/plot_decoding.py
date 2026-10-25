"""Context decodability of the best head in each layer. Fig. 3."""

import matplotlib.pyplot as plt
import torch

from smart_induction_heads.analysis import model_id
from smart_induction_heads.plotting import COLUMN_WIDTH, MODEL_COLORS, ORDER_COLORS, save

fig, ax = plt.subplots(1, len(MODEL_COLORS), figsize=(COLUMN_WIDTH, 1.35), sharey=True)
for i, model in enumerate(MODEL_COLORS):
    for order, color in ORDER_COLORS.items():
        accuracy = torch.load(f'data/decoding/markov{order}/{model_id(model)}.pt') * 100  # (splits, L, H)
        best = accuracy.mean(0).argmax(-1, keepdim=True)
        per_split = accuracy.gather(-1, best.expand(len(accuracy), -1, -1))[..., 0]  # (splits, L)
        mean, sem = per_split.mean(0), per_split.std(0) / len(per_split) ** 0.5
        ax[i].plot(mean, color=color, label=f'{order}{"nd" if order == 2 else "rd"} order')
        ax[i].fill_between(range(len(mean)), mean - sem, mean + sem, color=color, alpha=0.25, lw=0)
    ax[i].axhline(50, color='#8c8681', lw=0.6, ls=':')
    ax[i].set_ylim(45, 100)
    ax[i].set_xticks(range(0, per_split.size(1), 10))
    ax[i].set_title(model_id(model), fontweight='normal')
ax[0].set_ylabel('Context\ndecodability (%)')
ax[1].set_xlabel('Layer')
fig.legend(*ax[0].get_legend_handles_labels(), ncols=2, loc='lower center', bbox_to_anchor=(0.55, -0.04), handlelength=1.2)
fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.3)
save(fig, 'decoding')
