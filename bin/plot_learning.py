"""Next-token accuracy and context accuracy of the top induction and generic heads over chunks. Fig. 2A."""

import matplotlib.pyplot as plt

from smart_induction_heads.analysis import candidates, induction_scores, lm_accuracy, load_run, loo_top_heads
from smart_induction_heads.plotting import MODEL_COLORS, TEXT_WIDTH, plot_curve, save

fig, ax = plt.subplots(2, 3, figsize=(0.49 * TEXT_WIDTH, 2.95), sharey=True, sharex=True)
for row, order in enumerate([2, 3]):
    for model, color in MODEL_COLORS.items():
        model_accs, head_accs = load_run('none', order, model)
        curves = [lm_accuracy(model_accs, order)]
        for kind in ['induction', 'generic']: curves.append(loo_top_heads(head_accs, candidates(head_accs, induction_scores(model), kind), kind)[0])
        for col, curve in enumerate(curves): plot_curve(ax[row, col], curve, color=color, label=model.split('/')[-1])
for col, title in enumerate(['Next-token\nprediction', 'Induction\nheads', 'Generic ICL\nheads']): ax[0, col].set_title(title)
for row, order in enumerate(['2nd order', '3rd order']): ax[row, 0].set_ylabel(f'{order}\nAccuracy (%)')
for a in ax.flat:
    a.set_ylim(0, 100)
    a.set_xticks([0, 10, 20, 30])
fig.supxlabel('Chunks seen', y=0.08)
fig.legend(*ax[0, 0].get_legend_handles_labels(), ncols=3, loc='lower center', bbox_to_anchor=(0.5, -0.04), handlelength=1.2, columnspacing=1)
fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.6, w_pad=0.4)
save(fig, 'learning')
