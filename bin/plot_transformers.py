"""Small transformers trained from scratch: training loss, next-token accuracy and the best head's context accuracy. Fig. 7."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from smart_induction_heads.analysis import lm_accuracy, load_run
from smart_induction_heads.plotting import BLUE, COLUMN_WIDTH, RED, plot_curve, save

LAYER_COLORS = {4: RED, 2: BLUE}

runs = {json.loads((run / 'args.json').read_text())['n_layer']: run for run in Path('data/transformers').iterdir()}
fig, ax = plt.subplots(1, 3, figsize=(COLUMN_WIDTH, 1.6), layout='constrained')
for n_layer, color in LAYER_COLORS.items():
    run = runs[n_layer]
    config = json.loads((run / 'args.json').read_text())
    label = f'{n_layer} layers'
    loss = torch.tensor(json.loads((run / 'log.json').read_text())['loss'])
    tokens_per_step = config['batch_size'] * config['gradient_accumulation_steps'] * (config['block_size'] - 1)
    ax[0].plot(torch.arange(len(loss)) * 100 * tokens_per_step / 1e9, loss, color=color, lw=0.8, label=label)

    task = dict(chunk_size=config['chunk_size'], n_primitives=config['n_permute_primitive'])
    model_accs, head_accs = load_run('none', 3, run.name)
    plot_curve(ax[1], lm_accuracy(model_accs, 3, task), color=color, label=label)
    best = head_accs.mean((2, 3)).flatten().argmax()
    plot_curve(ax[2], head_accs.flatten(0, 1)[best], color=color, label=label)
ax[0].set_xlabel('Training tokens\n(billions)')
ax[0].set_ylabel('Loss')
ax[0].legend(loc='upper right', handlelength=1)
for col, title in [(0, 'Training'), (1, 'Next-token\nprediction'), (2, 'Best head')]: ax[col].set_title(title)
ax[1].set_ylabel('Accuracy (%)')
for col in (1, 2):
    ax[col].set_xlabel('Chunks seen')
    ax[col].set_ylim(0, 100)
    ax[col].set_xticks([0, 30, 60])
ax[2].tick_params(labelleft=False)
for col, letter, x in [(0, 'A', -0.35), (2, 'B', -0.12)]: ax[col].text(x, 1.3, letter, transform=ax[col].transAxes, fontweight='bold', fontsize=9)
save(fig, 'transformers')
