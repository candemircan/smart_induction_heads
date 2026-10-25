"""Attention of an adaptive and a static induction head from the last token of ambiguous prompts. Fig. 2D and App. A.9."""

import json
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from smart_induction_heads.plotting import BLUE, INK, RED, TEXT_WIDTH, save

parser = ArgumentParser()
parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B')
parser.add_argument('--adaptive', default='19-3', help='layer-head')
parser.add_argument('--static', default='2-3', help='layer-head')
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation='eager')
tokenizer = AutoTokenizer.from_pretrained(args.model)
heads = [tuple(map(int, h.split('-'))) for h in (args.adaptive, args.static)]
plt.rcParams.update({'mathtext.fontset': 'custom', 'mathtext.bf': 'Lato:bold', 'mathtext.it': 'Lato:italic'})
xticklabels = [f'Adaptive\nhead {args.adaptive}', f'Static\nhead {args.static}']


def last_token_attention(prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention from the last token of `prompt` for each head in `heads`.

    Returns:
        ids (torch.Tensor): token ids (T,)
        attn (torch.Tensor): attention per head (2, T)
    """
    ids = tokenizer(prompt, return_tensors='pt').input_ids
    with torch.no_grad(): attentions = model(ids, output_attentions=True).attentions
    return ids[0], torch.stack([attentions[l][0, h, -1] for l, h in heads]).float()


def bars(ax, correct, wrong, labels, errors=(None, None)):
    for k, (values, label, error, color) in enumerate([(correct, labels[0], errors[0], RED), (wrong, labels[1], errors[1], BLUE)]):
        ax.bar(torch.arange(2) + (k - 0.5) * 0.38, values, 0.36, yerr=error, color=color, label=label, ec=INK, lw=0.5, error_kw=dict(lw=0.8, ecolor=INK))
    ax.set_ylabel('Attention')
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticks(torch.arange(2), xticklabels)
    ax.tick_params(axis='x', length=0)


# Fig. 2D: summed attention to every occurrence of the correct and the wrong continuation
prompt = 'I visited San Antonio and saw the Alamo, and San Francisco where I saw the Golden Gate bridge. After seeing the Alamo I realized how much I liked San'
ids, attn = last_token_attention(prompt)
sums = [attn[:, torch.isin(ids, torch.tensor(tokenizer(word, add_special_tokens=False).input_ids))].sum(-1) for word in (' Antonio', ' Francisco')]
print(f'adaptive: Antonio {sums[0][0]:.3f}, Francisco {sums[1][0]:.3f}; static: Antonio {sums[0][1]:.3f}, Francisco {sums[1][1]:.3f}')
fig, ax = plt.subplots(figsize=(0.49 * TEXT_WIDTH, 2.3))
bars(ax, *sums, ['Antonio', 'Francisco'])
ax.set_ylim(0, 1.3)
ax.spines['left'].set_bounds(0, 1)
ax.legend(loc='upper right', bbox_to_anchor=(1, 0.74), handlelength=1)
text = ('I visited San $\\bf{Antonio}$ and saw the Alamo, and San $\\bf{Francisco}$\nwhere I saw the Golden Gate bridge. After seeing the\n'
        'Alamo I realized how much I liked $\\it{San}$ ...')
ax.text(0.02, 1.0, text, ha='left', va='top', transform=ax.transAxes, linespacing=1.4, bbox=dict(boxstyle='round,pad=0.5', fc='#ebe6df', ec='none'))
save(fig, 'language_example')

# App. A.9: mean attention to the first token of the correct and the wrong continuation, over all prompts
correct, wrong = [], []
for p in json.load(open('data/language.json')):
    ids, attn = last_token_attention(p['prompt'])
    correct.append(attn[:, ids == tokenizer.encode(p['c'])[0]].mean(-1))
    wrong.append(attn[:, ids == tokenizer.encode(p['w'])[0]].mean(-1))
correct, wrong = torch.stack(correct), torch.stack(wrong)  # (prompts, 2)
ci = lambda x: 1.96 * x.std(0) / len(x) ** 0.5
fig, ax = plt.subplots(figsize=(0.49 * TEXT_WIDTH, 2.0))
bars(ax, correct.mean(0), wrong.mean(0), ['Correct', 'Wrong'], (ci(correct), ci(wrong)))
ax.set_ylim(0, 1)
ax.legend(loc='upper right', handlelength=1)
save(fig, 'language_prompts')
