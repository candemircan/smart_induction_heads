#!/bin/bash
# All experiments of the paper, in dependency order. Needs a GPU.
set -euo pipefail

QWEN="Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B"
OTHERS="google/gemma-2-2b meta-llama/Llama-3.2-3B HuggingFaceTB/SmolLM3-3B Qwen/Qwen3-0.6B"

for model in $QWEN $OTHERS; do
    uv run bin/score_induction.py --model "$model"
    for order in 2 3; do
        uv run bin/decode_context.py --model "$model" --order "$order"
        uv run bin/evaluate.py --model "$model" --order "$order"
    done
done

for model in $QWEN; do
    for order in 2 3; do
        uv run bin/evaluate.py --model "$model" --order "$order" --ablate induction
        uv run bin/evaluate.py --model "$model" --order "$order" --ablate random-induction
        uv run bin/evaluate.py --model "$model" --order "$order" --ablate context
        uv run bin/evaluate.py --model "$model" --order "$order" --ablate random-context --matched
        uv run bin/evaluate.py --model "$model" --order "$order" --chunk_sizes $([ "$order" = 2 ] && echo 4 12 || echo 3 5)
        for noise in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4; do
            uv run bin/evaluate.py --model "$model" --order "$order" --noise "$noise"
        done
    done
done

for model in $OTHERS; do
    for order in 2 3; do
        uv run bin/evaluate.py --model "$model" --order "$order" --ablate context
        uv run bin/evaluate.py --model "$model" --order "$order" --ablate random-context
    done
done

for layers in "2 1280 16" "4 896 8"; do
    read -r n_layer n_embd n_head <<< "$layers"
    uv run bin/train_transformer.py --n_layer "$n_layer" --n_embd "$n_embd" --n_head "$n_head"
done
for run in data/transformers/*/; do
    uv run bin/evaluate_transformer.py --run "$run"
done
