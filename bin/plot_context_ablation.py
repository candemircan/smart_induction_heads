"""Next-token accuracy and induction-head context accuracy with context matching heads or as many layer-matched random heads ablated. Fig. 5."""

import matplotlib.pyplot as plt
from einops import rearrange

from smart_induction_heads.analysis import candidates, induction_scores, lm_accuracy, load_run, top_heads
from smart_induction_heads.plotting import ABLATION_COLORS, MODEL_COLORS, TEXT_WIDTH, plot_curve, save

CONDITIONS = [('none', 'No ablation'), ('random-context-matched', 'Random heads ablated'), ('context', 'Context matching heads ablated')]

fig = plt.figure(figsize=(TEXT_WIDTH, 2.5), layout='constrained')
panels = fig.subfigures(1, 2, wspace=0.03)
ax = [panel.subplots(2, 3, sharex=True, sharey=True) for panel in panels]
for panel, title in zip(panels, ['Next-token prediction', 'Induction heads']): panel.suptitle(title, fontweight='bold')
for col, model in enumerate(MODEL_COLORS):
    for row, order in enumerate([2, 3]):
        for (run, label), color in zip(CONDITIONS, ABLATION_COLORS.values()):
            model_accs, head_accs = load_run(run, order, model)
            plot_curve(ax[0][row, col], lm_accuracy(model_accs, order), label=label, color=color)
            heads = top_heads(head_accs, candidates(head_accs, induction_scores(model), 'induction'))
            plot_curve(ax[1][row, col], rearrange(heads, 'k b c -> (k b) c'), label=label, color=color)
    for a in ax: a[0, col].set_title(model.split('/')[-1], fontweight='normal')
for row, order in enumerate(['2nd order', '3rd order']): ax[0][row, 0].set_ylabel(f'{order}\nAccuracy (%)')
for a in ax:
    for b in a.flat:
        b.set_ylim(0, 100)
        b.set_xticks([0, 10, 20, 30])
    a[1, 1].set_xlabel('Chunks seen')
fig.legend(*ax[0][0, 0].get_legend_handles_labels(), ncols=3, loc='outside lower center', handlelength=1.2)
save(fig, 'context_ablation')
