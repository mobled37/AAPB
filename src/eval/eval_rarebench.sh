#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

EVAL_SCRIPT="src/eval/eval_by_GPT.py"

# Set the base image directory for the experiment
IMAGE_DIR="output"

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

echo "=============================="
echo "Evaluating: ${IMAGE_DIR}"
echo "=============================="
for cat in "${CATEGORIES[@]}"; do
    python ${EVAL_SCRIPT} --input_dir "${IMAGE_DIR}/${cat}"
done
