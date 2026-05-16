#!/usr/bin/env bash

# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

GPUS="${LLADAO_GENEVAL_GPUS:-8}"
MASTER_PORT="${LLADAO_GENEVAL_MASTER_PORT:-29502}"
METADATA_FILE="${GENEVAL_METADATA_FILE:-./eval/gen/geneval/prompts/evaluation_metadata_long.jsonl}"
NUM_IMAGES="${GENEVAL_NUM_IMAGES:-4}"
BATCH_SIZE="${GENEVAL_BATCH_SIZE:-1}"
RESOLUTION="${GENEVAL_RESOLUTION:-1024}"
MAX_LATENT_SIZE="${GENEVAL_MAX_LATENT_SIZE:-64}"
SKIP_GENERATION="${GENEVAL_SKIP_GENERATION:-0}"
DETECTOR_MODEL_PATH="${GENEVAL_DETECTOR_MODEL_PATH:-${PROJECT_ROOT}/eval/gen/geneval/evaluation/models/mask2former}"
CLIP_PRETRAINED="${GENEVAL_CLIP_PRETRAINED:-laion2b_s32b_b82k}"

MODEL_PATHS=()
if [ "$#" -gt 0 ]; then
    MODEL_PATHS=("$@")
elif [ -n "${LLADAO_MODEL_PATHS:-}" ]; then
    # shellcheck disable=SC2206
    MODEL_PATHS=(${LLADAO_MODEL_PATHS})
elif [ -n "${LLADAO_MODEL_PATH:-}" ]; then
    MODEL_PATHS=("${LLADAO_MODEL_PATH}")
else
    echo "Please pass model paths as arguments, or set LLADAO_MODEL_PATH / LLADAO_MODEL_PATHS." >&2
    exit 1
fi

if [ ! -f "${DETECTOR_MODEL_PATH}/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth" ]; then
    echo "Geneval detector weights are missing under ${DETECTOR_MODEL_PATH}." >&2
    echo "Set GENEVAL_DETECTOR_MODEL_PATH or run:" >&2
    echo "bash eval/gen/geneval/evaluation/download_models.sh ${DETECTOR_MODEL_PATH}" >&2
    exit 1
fi

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    echo "========================================="
    echo "Processing: ${MODEL_PATH}"
    echo "GenEval detector path: ${DETECTOR_MODEL_PATH}"
    echo "OpenCLIP pretrained: ${CLIP_PRETRAINED}"
    echo "========================================="

    OUTPUT_DIR="${MODEL_PATH}/gen_eval_images"
    RESULT_FILE="${MODEL_PATH}/geneval_results_long.jsonl"

    USE_REG=0
    if [[ "${MODEL_PATH}" =~ variant[0-9]+_2(/|$) ]]; then
        USE_REG=1
        echo "Detected variant_2 model, enabling --reg."
    fi

    if [ "${SKIP_GENERATION}" != "1" ]; then
        echo "Step 1: Generating images..."
        if [ "${USE_REG}" = "1" ]; then
            torchrun \
                --nnodes=1 \
                --node_rank=0 \
                --nproc_per_node="${GPUS}" \
                --master_addr=127.0.0.1 \
                --master_port="${MASTER_PORT}" \
                ./eval/gen/gen_images_mp_dllm.py \
                --output_dir "${OUTPUT_DIR}" \
                --metadata_file "${METADATA_FILE}" \
                --batch_size "${BATCH_SIZE}" \
                --num_images "${NUM_IMAGES}" \
                --resolution "${RESOLUTION}" \
                --max_latent_size "${MAX_LATENT_SIZE}" \
                --model-path "${MODEL_PATH}" \
                --reg
        else
            torchrun \
                --nnodes=1 \
                --node_rank=0 \
                --nproc_per_node="${GPUS}" \
                --master_addr=127.0.0.1 \
                --master_port="${MASTER_PORT}" \
                ./eval/gen/gen_images_mp_dllm.py \
                --output_dir "${OUTPUT_DIR}" \
                --metadata_file "${METADATA_FILE}" \
                --batch_size "${BATCH_SIZE}" \
                --num_images "${NUM_IMAGES}" \
                --resolution "${RESOLUTION}" \
                --max_latent_size "${MAX_LATENT_SIZE}" \
                --model-path "${MODEL_PATH}"
        fi
    fi

    echo "Step 2: Calculating scores..."
    torchrun \
        --nnodes=1 \
        --node_rank=0 \
        --nproc_per_node="${GPUS}" \
        --master_addr=127.0.0.1 \
        --master_port="${MASTER_PORT}" \
        ./eval/gen/geneval/evaluation/evaluate_images_mp.py \
        "${OUTPUT_DIR}" \
        --outfile "${RESULT_FILE}" \
        --model-path "${DETECTOR_MODEL_PATH}" \
        --options "clip_pretrained=${CLIP_PRETRAINED}"

    python ./eval/gen/geneval/evaluation/summary_scores.py "${RESULT_FILE}"

    echo "Completed: ${MODEL_PATH}"
    echo
done

echo "========================================="
echo "All models processed!"
echo "========================================="
