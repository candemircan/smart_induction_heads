#!/bin/bash
# All figures and statistics of the paper, from the results of bin/experiments.sh.
# The scripts marked "runs a model" load a language model; the rest read data/ only.
set -euo pipefail
mkdir -p figures

uv run bin/plot_overview.py                             # Fig. 1, runs a model
uv run bin/plot_prompt_example.py                       # App. A.7
uv run bin/plot_learning.py                             # Fig. 2A
uv run bin/plot_attention_maps.py                       # Fig. 2B, runs a model
uv run bin/plot_induction_ablation.py                   # Fig. 2C
uv run bin/plot_language.py                             # Fig. 2D and App. A.9, runs a model
uv run bin/plot_decoding.py                             # Fig. 3
for order in 2 3; do                                    # Fig. 4, runs a model
    uv run bin/plot_attention_maps.py --order "$order" --heads 13-4 14-8 \
        --titles "Off-diagonal head" "N-tokens-back head" --out "context_heads_order${order}"
done
uv run bin/plot_context_ablation.py                     # Fig. 5
uv run bin/plot_ablated_head.py                         # Fig. 6, runs a model
uv run bin/plot_transformers.py                         # Fig. 7
uv run bin/plot_variable_chunks.py                      # App. A.5
uv run bin/plot_depth.py                                # App. A.8
uv run bin/plot_head_stability.py                       # App. B.1
uv run bin/plot_noise.py                                # App. B.2
for model in google/gemma-2-2b meta-llama/Llama-3.2-3B HuggingFaceTB/SmolLM3-3B Qwen/Qwen3-0.6B; do
    uv run bin/plot_replication.py --model "$model"     # App. C
done
