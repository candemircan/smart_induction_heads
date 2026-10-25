"""Next-token accuracy and context accuracy of the top induction heads with a chunk size drawn per sequence. App. A.5."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from smart_induction_heads.analysis import candidates, induction_scores, loo_top_heads, model_id
from smart_induction_heads.plotting import MODEL_COLORS, TEXT_WIDTH, plot_curve, save

RUNS = {2: 'none-chunks4-12', 3: 'none-chunks3-5'}

fig, ax = plt.subplots(2, 2, figsize=(0.62 * TEXT_WIDTH, 2.9), sharex=True, sharey=True, layout='constrained')
for row, (order, name) in enumerate(RUNS.items()):
    for model, color in MODEL_COLORS.items():
        path = Path('data/runs') / name / f'markov{order}' / model_id(model)
        head_accs = torch.load(path / 'head_accs.pt')
        plot_curve(ax[row, 0], torch.load(path / 'lm_accs.pt'), color=color, label=model_id(model))
        plot_curve(ax[row, 1], loo_top_heads(head_accs, candidates(head_accs, induction_scores(model), 'induction'), 'induction')[0], color=color)
    ax[row, 0].set_ylabel(f'{order}{"nd" if order == 2 else "rd"} order\nAccuracy (%)')
for col, title in enumerate(['Next-token prediction', 'Induction heads']):
    ax[0, col].set_title(title)
    ax[1, col].set_xlabel('Chunks seen')
for a in ax.flat:
    a.set_ylim(0, 100)
    a.set_xticks([0, 10, 20, 30])
fig.legend(*ax[0, 0].get_legend_handles_labels(), ncols=3, loc='outside lower center', handlelength=1.2)
save(fig, 'variable_chunks')
