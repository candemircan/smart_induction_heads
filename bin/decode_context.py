"""Decode from each head's output whether the previous chunk had the same chunk id as the current one (Sec. 5, App. A.6).

The features are a head's outputs (Eq. 3) averaged over the tokens of a chunk. An L2-regularised
logistic regression is fit on 75% of the chunks and scored by balanced accuracy on the rest,
for several random splits. Saves the accuracies (splits, L, H) to data/decoding/markov<order>/<model>.pt.
"""

import warnings
from argparse import ArgumentParser
from pathlib import Path

import torch
from einops import rearrange
from nnsight import LanguageModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from smart_induction_heads.analysis import model_id
from smart_induction_heads.sequences import TASKS, sample

warnings.filterwarnings('ignore')

parser = ArgumentParser()
parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B')
parser.add_argument('--order', type=int, default=2, choices=[2, 3])
parser.add_argument('--n_seqs', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_splits', type=int, default=10)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

torch.manual_seed(args.seed)
model = LanguageModel(args.model, device_map='auto', torch_dtype=torch.bfloat16, attn_implementation='eager', dispatch=True)
L, H = model.config.num_hidden_layers, model.config.num_attention_heads
task = TASKS[args.order]
chunk_len = task['chunk_size'] * task['n_primitives']
batches = [sample(args.batch_size, torch.arange(model.config.vocab_size), **task) for _ in range(args.n_seqs // args.batch_size)]

features, labels = [], []
for seqs, ids in batches:
    with torch.no_grad(), model.trace(seqs.to(model.device), scan=False):
        outputs = [layer.self_attn.o_proj.input.save() for layer in model.model.layers]
    # chunk means of each head's output, without the first chunk, which has no predecessor
    x = torch.stack([rearrange(o.value, 'b (c t) (h d) -> b c t h d', t=chunk_len, h=H).mean(2) for o in outputs])
    features.append(rearrange(x[:, :, 1:].float().cpu(), 'l b c h d -> l h (b c) d'))
    labels.append((ids[:, 1:] == ids[:, :-1]).flatten())
features, labels = torch.cat(features, 2).numpy(), torch.cat(labels).float().numpy()

accuracy = torch.zeros(args.n_splits, L, H)
for split in range(args.n_splits):
    for layer in range(L):
        for head in range(H):
            X_train, X_test, y_train, y_test = train_test_split(features[layer, head], labels, test_size=0.25, random_state=args.seed + split, stratify=labels)
            classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_train, y_train)
            accuracy[split, layer, head] = balanced_accuracy_score(y_test, classifier.predict(X_test))

Path(f'data/decoding/markov{args.order}').mkdir(parents=True, exist_ok=True)
torch.save(accuracy, f'data/decoding/markov{args.order}/{model_id(args.model)}.pt')
