"""How often each head is among the top 5 across the leave-one-out folds. App. B.1."""

import matplotlib.pyplot as plt
import numpy as np
import torch

from smart_induction_heads.analysis import candidates, induction_scores, load_run, loo_top_heads
from smart_induction_heads.plotting import ACCENT, BLUE, GREY, INK, MODEL_COLORS, RED, TEXT_WIDTH, save

MODELS = list(MODEL_COLORS)
STATS = [('Unique heads', BLUE), ('Mean frequency (%)', RED), ('Min frequency (%)', ACCENT), ('Max frequency (%)', GREY)]

fig, ax = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 3.6), sharey=True)
for row, order in enumerate([2, 3]):
    for col, kind in enumerate(['induction', 'generic']):
        stats = []
        for model in MODELS:
            _, head_accs = load_run('none', order, model)
            selected = loo_top_heads(head_accs, candidates(head_accs, induction_scores(model), kind), kind)[1]
            _, counts = selected[selected >= 0].unique(return_counts=True)
            freq = counts / len(selected) * 100
            stats.append([len(counts), freq.mean(), freq.min(), freq.max()])
        stats = torch.tensor(stats).T  # (stat, model)
        x, width = np.arange(len(MODELS)), 0.2
        for i, ((label, color), values) in enumerate(zip(STATS, stats)):
            bars = ax[row, col].bar(x + (i - 1.5) * width, values, width, label=label, color=color, ec=INK, lw=0.4)
            ax[row, col].bar_label(bars, labels=[f'{v:.0f}' for v in values], padding=1, fontsize=5)
        ax[row, col].set_xticks(x, [m.split('/')[-1] for m in MODELS])
        ax[row, col].set_ylim([0, stats.max() * 1.15])
        ax[row, col].set_title(f"{order}{'nd' if order == 2 else 'rd'} order, {'induction' if kind == 'induction' else 'generic ICL'} heads")
    ax[row, 0].set_ylabel('Count or frequency (%)')
fig.legend(*ax[0, 0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, handlelength=1)
plt.tight_layout(rect=[0, 0, 1, 0.95])
save(fig, 'head_stability')
