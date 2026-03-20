#!/bin/bash
# Generate R2F anchor prompts for RareBench categories using GPT-4o.
#
# Usage:
#   cd src/gpt
#   bash get_r2f_response_from_GPT.sh

PROMPT_DIR="../../dataset/original_prompt/rarebench"
OUT_DIR="../../dataset/r2f_prompt/rarebench"

CATEGORIES=(
    rarebench_single_1property
    rarebench_single_2shape
    rarebench_single_3texture
    rarebench_single_4action
    rarebench_single_5complex
    rarebench_multi_1and
    rarebench_multi_2relation
    rarebench_multi_3complex
)

for cat in "${CATEGORIES[@]}"; do
    echo ">>> ${cat}"
    python get_r2f_response_from_GPT.py \
        --test_file "${PROMPT_DIR}/${cat}.txt" \
        --out_dir "${OUT_DIR}"
done
