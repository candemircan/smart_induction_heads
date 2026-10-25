"""Accuracy as a function of the probability of replacing each token. App. B.2."""

import matplotlib.pyplot as plt
import numpy as np
import torch

from smart_induction_heads.analysis import candidates, induction_scores, lm_accuracy, load_run, loo_top_heads
from smart_induction_heads.plotting import MODEL_COLORS, TEXT_WIDTH, save

NOISE = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]


def run(order, model, noise): return load_run('none' + f'-noise{noise}' * (noise > 0), order, model)


def most_selected(head_accs, mask, kind, k=5):
    """Flat indices of the `k` heads most often among the top heads across leave-one-out folds."""
    selected = loo_top_heads(head_accs, mask, kind)[1]
    heads, counts = selected[selected >= 0].unique(return_counts=True)
    return heads[counts.sort(descending=True, stable=True).indices[:k]]


fig, ax = plt.subplots(2, 3, figsize=(TEXT_WIDTH, 3.2), sharey=True, sharex=True)
x = np.arange(len(NOISE))
for row, order in enumerate([2, 3]):
    for model, color in MODEL_COLORS.items():
        # each panel shows the mean and standard error over its values at each noise level
        values = [[lm_accuracy(run(order, model, n)[0], order).flatten() for n in NOISE]]
        _, clean = run(order, model, 0.0)
        for kind in ['induction', 'generic']:
            heads = most_selected(clean, candidates(clean, induction_scores(model), kind), kind)
            values.append([run(order, model, n)[1].flatten(0, 1)[heads].mean((0, 2)) for n in NOISE])
        for col, per_noise in enumerate(values):
            mean = torch.stack([v.mean() for v in per_noise]) * 100
            sem = torch.stack([v.std() / len(v) ** 0.5 for v in per_noise]) * 100
            ax[row, col].plot(x, mean, marker='o', ms=2.5, color=color, label=model.split('/')[-1])
            ax[row, col].fill_between(x, mean - sem, mean + sem, color=color, alpha=0.25, lw=0)
            ax[row, col].set_ylim([0, 100])
            ax[row, col].set_xticks(x, [f'{n:.2f}' if n > 0 else '0' for n in NOISE], rotation=45, ha='right')
for col, title in enumerate(['Next-token prediction', 'Induction heads', 'Generic ICL heads']): ax[0, col].set_title(title)
ax[0, 0].set_ylabel('2nd order\nAccuracy (%)')
ax[1, 0].set_ylabel('3rd order\nAccuracy (%)')
fig.legend(*ax[1, -1].get_legend_handles_labels(), ncols=3, loc='upper center', bbox_to_anchor=(0.5, 0.01))
fig.supxlabel('Probability of replacing a token', y=0.06)
plt.tight_layout(rect=[0, 0.05, 1, 1])
save(fig, 'noise')
