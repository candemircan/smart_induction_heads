"""Next-token accuracy with induction heads or as many random heads ablated. Fig. 2C."""

import matplotlib.pyplot as plt

from smart_induction_heads.analysis import lm_accuracy, load_run
from smart_induction_heads.plotting import ABLATION_COLORS, MODEL_COLORS, TEXT_WIDTH, plot_curve, save

CONDITIONS = [('none', 'No ablation'), ('random-induction', 'Random heads ablated'), ('induction', 'Induction heads ablated')]

fig, ax = plt.subplots(2, 3, figsize=(0.49 * TEXT_WIDTH, 2.3), sharey=True, sharex=True)
for col, model in enumerate(MODEL_COLORS):
    for row, order in enumerate([2, 3]):
        for (run, label), color in zip(CONDITIONS, ABLATION_COLORS.values()):
            plot_curve(ax[row, col], lm_accuracy(load_run(run, order, model)[0], order), label=label, color=color)
    ax[0, col].set_title(model.split('/')[-1])
for row, order in enumerate(['2nd order', '3rd order']): ax[row, 0].set_ylabel(f'{order}\nAccuracy (%)')
for a in ax.flat:
    a.set_ylim(0, 100)
    a.set_xticks([0, 10, 20, 30])
fig.supxlabel('Chunks seen', y=0.08)
fig.legend(*ax[0, 0].get_legend_handles_labels(), ncols=3, loc='lower center', bbox_to_anchor=(0.5, -0.04), handlelength=1.2, columnspacing=1)
fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.6, w_pad=0.4)
save(fig, 'induction_ablation')
