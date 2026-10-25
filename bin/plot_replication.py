"""Context matching ablation and context decodability for one model. App. C."""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
from einops import rearrange

from smart_induction_heads.analysis import candidates, induction_scores, lm_accuracy, load_run, model_id, top_heads
from smart_induction_heads.plotting import ABLATION_COLORS, GREY, ORDER_COLORS, TEXT_WIDTH, plot_curve, save

parser = ArgumentParser()
parser.add_argument('--model', default='google/gemma-2-2b')
args = parser.parse_args()

CONDITIONS = [('none', 'No ablation'), ('random-context', 'Random heads ablated'), ('context', 'Context matching heads ablated')]

fig, ax = plt.subplots(2, 3, figsize=(TEXT_WIDTH, 2.6), sharex='col', layout='constrained')
for row, order in enumerate([2, 3]):
    for (run, label), color in zip(CONDITIONS, ABLATION_COLORS.values()):
        model_accs, head_accs = load_run(run, order, args.model)
        plot_curve(ax[row, 0], lm_accuracy(model_accs, order), label=label, color=color)
        heads = top_heads(head_accs, candidates(head_accs, induction_scores(args.model), 'induction'))
        plot_curve(ax[row, 1], rearrange(heads, 'k b c -> (k b) c'), label=label, color=color)
    decoding = torch.load(f'data/decoding/markov{order}/{model_id(args.model)}.pt')[0]
    ax[row, 2].plot(decoding.max(-1).values * 100, color=ORDER_COLORS[order])
    ax[row, 2].axhline(50, color=GREY, lw=0.6, ls=':')
    ax[row, 2].set_ylim([45, 100])
    for col in range(2): ax[row, col].set_ylim([0, 100])
    ax[row, 0].set_ylabel(f'{order}{"nd" if order == 2 else "rd"} order\nAccuracy (%)')
    ax[row, 2].set_ylabel('Context\ndecodability (%)')
for col, title in enumerate(['Next-token prediction', 'Induction heads', 'Context decodability']): ax[0, col].set_title(title)
for col, label in enumerate(['Chunks seen', 'Chunks seen', 'Layer']): ax[1, col].set_xlabel(label)
fig.legend(*ax[0, 0].get_legend_handles_labels(), ncols=3, loc='outside lower center', handlelength=1.2)
fig.suptitle(model_id(args.model), fontweight='bold')
save(fig, f'replication_{model_id(args.model)}')
