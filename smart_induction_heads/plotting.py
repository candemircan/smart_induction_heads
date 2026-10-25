"""Shared plot elements."""

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap

from .analysis import mean_sem

# limonata Helgoland (nature) palette, https://github.com/sarraahr/limonata-colors
RED, BLUE, ACCENT, GREY, INK = '#962100', '#5e7b99', '#cf9a4a', '#8c8681', '#262626'
ORDER_COLORS = {2: RED, 3: ACCENT}
MODEL_COLORS = {'Qwen/Qwen2.5-0.5B': '#9cc5b5', 'Qwen/Qwen2.5-1.5B': '#4f9483', 'Qwen/Qwen2.5-3B': '#1d5547'}
ABLATION_COLORS = {'none': RED, 'random': GREY, 'target': BLUE}
ATTENTION_CMAP = LinearSegmentedColormap.from_list('attention', ['white', RED])

# ACL page widths in inches
TEXT_WIDTH, COLUMN_WIDTH = 6.3, 3.03


def plot_curve(ax, values: torch.Tensor, **kwargs):
    """Plot the mean of `values` (n, C) in percent, with the standard error as a band."""
    mean, sem = mean_sem(values * 100)
    line, = ax.plot(mean, **kwargs)
    ax.fill_between(range(len(mean)), mean - sem, mean + sem, color=line.get_color(), alpha=0.25, lw=0)


def save(fig, name: str):
    """Save `fig` as figures/<name>.svg and figures/<name>.pdf."""
    for suffix in ('svg', 'pdf'): fig.savefig(f'figures/{name}.{suffix}', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
