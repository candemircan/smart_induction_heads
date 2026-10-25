"""Example 2nd- and 3rd-order prompts, one chunk per line. App. A.7.

2nd order: the box colour marks the chunk. 3rd order: the box colour marks the 2nd-order chunk, and the label on
the left the 3rd-order chunk.
"""

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import FancyBboxPatch
from transformers import AutoConfig, AutoTokenizer

from smart_induction_heads.plotting import ACCENT, BLUE, GREY, INK, RED, save
from smart_induction_heads.sequences import second_order, third_order

MODEL = 'Qwen/Qwen2.5-1.5B'
COLORS = [RED, BLUE, ACCENT]
LETTERS = ['$\\Omega$', '$\\psi$', '$\\phi$']
chunk_size, n_chunks, n_primitives, n_reps = 3, 3, 3, 3

# some random tokens are Chinese, which Lato does not cover
plt.rcParams['font.family'] = ['Lato', 'Hiragino Sans GB']
tokenizer = AutoTokenizer.from_pretrained(MODEL)
vocab_size = AutoConfig.from_pretrained(MODEL).vocab_size


def draw(ax, x0, y0, words, colors, per_line, labels=None):
    """Draw `words` as coloured boxes, `per_line` to a line, from the top left corner (x0, y0) in points."""
    for i, (word, color) in enumerate(zip(words, colors)):
        x, y = x0 + (i % per_line) * (BOX + GAP), y0 - (i // per_line) * LINE
        ax.add_patch(FancyBboxPatch((x, y - HEIGHT), BOX, HEIGHT, boxstyle='round,pad=0,rounding_size=1.5', fc=color, ec='none'))
        ax.text(x + BOX / 2, y - HEIGHT / 2, word.strip(), ha='center', va='center', fontsize=4.6, color=INK if color == ACCENT else 'white')
    if labels is not None:
        for line, label in enumerate(labels): ax.text(x0 - 3, y0 - line * LINE - HEIGHT / 2, label, ha='right', va='center', fontsize=6, color=GREY)


examples = []
for example in range(2):
    torch.manual_seed(example + 123456)
    tokens = torch.randint(vocab_size, (chunk_size,))
    seq, ids, primitive_ids = third_order(tokens, n_chunks, n_primitives, n_reps)
    third = (tokenizer.batch_decode(seq), [COLORS[i] for i in primitive_ids.flatten().repeat_interleave(chunk_size)], [LETTERS[i] for i in ids])
    seq, ids = second_order(tokens, n_chunks, n_reps)
    examples.append(((tokenizer.batch_decode(seq), [COLORS[i] for i in ids.repeat_interleave(chunk_size)]), third))

# sizes in points
BOX, GAP, HEIGHT, LINE = 33, 3, 9, 11
left, block = 3 * (BOX + GAP) + 30, n_chunks * n_reps * LINE + 16
width, height = left + chunk_size * n_primitives * (BOX + GAP), 2 * block + 12
fig = plt.figure(figsize=(width / 72, height / 72))
ax = fig.add_axes((0, 0, 1, 1), xlim=(0, width), ylim=(0, height))
ax.axis('off')
ax.text(1.5 * (BOX + GAP), height - 2, '2nd order', ha='center', va='top', fontweight='bold')
ax.text(left + 4.5 * (BOX + GAP), height - 2, '3rd order', ha='center', va='top', fontweight='bold')
for row, (second, third) in enumerate(examples):
    top = height - 14 - row * block
    draw(ax, 0, top, *second, chunk_size)
    draw(ax, left, top, *third[:2], chunk_size * n_primitives, third[2])
save(fig, 'prompt_example')
