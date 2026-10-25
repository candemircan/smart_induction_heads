"""Context accuracy of induction heads against their relative depth, with Spearman correlations. App. A.8."""

import matplotlib.pyplot as plt
import torch
from scipy.stats import spearmanr

from smart_induction_heads.analysis import induction_scores, load_run
from smart_induction_heads.heads import INDUCTION_THRESHOLD
from smart_induction_heads.plotting import MODEL_COLORS, TEXT_WIDTH, save


fig, ax = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.6), layout='constrained')
for col, order in enumerate([2, 3]):
    depths, scores = [], []
    for model, color in MODEL_COLORS.items():
        _, head_accs = load_run('none', order, model)
        layer, head = (induction_scores(model) > INDUCTION_THRESHOLD).nonzero().T
        # mean context accuracy over the last 10 chunks
        score = head_accs[layer, head].mean(1)[:, -10:].mean(-1)
        depth = layer / head_accs.size(0)
        ax[col].scatter(depth, score, s=12, color=color, label=model.split('/')[-1])
        depths.append(depth)
        scores.append(score)
    rho, p = spearmanr(torch.cat(depths), torch.cat(scores))
    print(f'markov{order}: rho={rho:.3f}, p={p:.3e}')
    ax[col].set_title(f'{order}{"nd" if order == 2 else "rd"} order')
    ax[col].set_ylim(0, 1.02)
    ax[col].set_xlabel('Relative depth')
    ax[col].set_ylabel('Learning score')
fig.legend(*ax[0].get_legend_handles_labels(), loc='outside upper center', ncol=3)
save(fig, 'depth')
