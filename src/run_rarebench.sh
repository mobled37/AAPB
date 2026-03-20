#!/bin/bash
# Run AAPB inference on all 8 RareBench categories.
#
# Usage:
#   bash src/run_rarebench.sh
#   CUDA_VISIBLE_DEVICES=1 bash src/run_rarebench.sh

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

CATEGORIES=(
    rarebench_single_1property_gpt4.txt
    rarebench_single_2shape_gpt4.txt
    rarebench_single_3texture_gpt4.txt
    rarebench_single_4action_gpt4.txt
    rarebench_single_5complex_gpt4.txt
    rarebench_multi_1and_gpt4.txt
    rarebench_multi_2relation_gpt4.txt
    rarebench_multi_3complex_gpt4.txt
)

SEED=${SEED:-42}
OUT_PATH="${PROJECT_DIR}/output/"

echo "=============================="
echo "AAPB RareBench Inference"
echo "Output: ${OUT_PATH}"
echo "Seed: ${SEED}"
echo "=============================="

for cat in "${CATEGORIES[@]}"; do
    echo ""
    echo ">>> ${cat}"
    python "${SCRIPT_DIR}/inference.py" \
        --test_file "${PROJECT_DIR}/dataset/r2f_prompt/rarebench/${cat}" \
        --out_path "${OUT_PATH}" \
        --seed "${SEED}"
done

echo ""
echo "Done. Results saved to ${OUT_PATH}"
