#!/usr/bin/env bash

# Copyright 2025 AntGroup and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

TARGET_PATH="${1:-${LLADAO_MODEL_PATH:-}}"
METADATA_FILE="${2:-${DPG_METADATA_FILE:-}}"
OUTPUT_DIR="${3:-${DPG_OUTPUT_DIR:-}}"

SKIP_GENERATION="${DPG_SKIP_GENERATION:-0}"
SKIP_GRID="${DPG_SKIP_GRID:-0}"
SKIP_SCORE="${DPG_SKIP_SCORE:-0}"

if [ -z "${TARGET_PATH}" ]; then
    echo "Usage: bash scripts/eval/run_dpg_dllm.sh <model_path_or_raw_image_root> <metadata_file> [output_dir]" >&2
    echo "You can also set LLADAO_MODEL_PATH / DPG_METADATA_FILE / DPG_OUTPUT_DIR." >&2
    exit 1
fi

if [ "${SKIP_GENERATION}" != "1" ] && [ -z "${METADATA_FILE}" ]; then
    echo "Missing DPG metadata file. Pass it as the second argument or set DPG_METADATA_FILE." >&2
    exit 1
fi

if [ "${SKIP_GENERATION}" != "1" ] && [ ! -f "${METADATA_FILE}" ]; then
    echo "DPG metadata file not found: ${METADATA_FILE}" >&2
    exit 1
fi

MODEL_PATH=""
RAW_OUTPUT_DIR=""
if [ "${SKIP_GENERATION}" = "1" ]; then
    RAW_OUTPUT_DIR="${DPG_RAW_OUTPUT_DIR:-}"
    if [ -z "${RAW_OUTPUT_DIR}" ]; then
        if [ -n "${OUTPUT_DIR}" ]; then
            RAW_OUTPUT_DIR="${OUTPUT_DIR}"
        else
            RAW_OUTPUT_DIR="${TARGET_PATH}"
        fi
    fi
else
    MODEL_PATH="${TARGET_PATH}"
    if [ -z "${OUTPUT_DIR}" ]; then
        RAW_OUTPUT_DIR="${MODEL_PATH}/dpg_eval_images"
    else
        RAW_OUTPUT_DIR="${OUTPUT_DIR}"
    fi
fi

GPUS="${LLADAO_DPG_GPUS:-8}"
MASTER_PORT="${LLADAO_DPG_MASTER_PORT:-29512}"
NUM_IMAGES="${DPG_NUM_IMAGES:-4}"
BATCH_SIZE="${DPG_BATCH_SIZE:-1}"
RESOLUTION="${DPG_RESOLUTION:-1024}"
MAX_LATENT_SIZE="${DPG_MAX_LATENT_SIZE:-64}"
GRID_OUTPUT_DIR="${DPG_GRID_OUTPUT_DIR:-${RAW_OUTPUT_DIR}_grid}"
DPG_CSV_PATH="${DPG_CSV_PATH:-./eval/gen/dpg_bench/dpg_bench.csv}"
DPG_SCORE_PIC_NUM="${DPG_SCORE_PIC_NUM:-4}"
DPG_GRID_WORKERS="${DPG_GRID_WORKERS:-8}"
DPG_GRID_FORMAT="${DPG_GRID_FORMAT:-png}"
DPG_GRID_PNG_COMPRESS_LEVEL="${DPG_GRID_PNG_COMPRESS_LEVEL:-1}"
DPG_GRID_JPEG_QUALITY="${DPG_GRID_JPEG_QUALITY:-95}"
DPG_GRID_OVERWRITE="${DPG_GRID_OVERWRITE:-0}"
DPG_SCORE_PROCESSES="${DPG_SCORE_PROCESSES:-8}"
DPG_SCORE_PORT="${DPG_SCORE_PORT:-29520}"
DPG_SCORE_VQA_MODEL="${DPG_SCORE_VQA_MODEL:-mplug}"
DPG_SCORE_VQA_MODEL_PATH="${DPG_SCORE_VQA_MODEL_PATH:-damo/mplug_visual-question-answering_coco_large_en}"

SCORE_IMAGE_ROOT="${GRID_OUTPUT_DIR}"
if [ "${SKIP_GRID}" = "1" ]; then
    SCORE_IMAGE_ROOT="${RAW_OUTPUT_DIR}"
fi
DPG_SCORE_RESULT_PATH="${DPG_SCORE_RESULT_PATH:-${SCORE_IMAGE_ROOT}/dpg-bench_results.txt}"

if [ ! -f "${DPG_CSV_PATH}" ]; then
    echo "DPG CSV not found: ${DPG_CSV_PATH}" >&2
    exit 1
fi

if [ "${SKIP_GENERATION}" = "1" ] && [ ! -d "${RAW_OUTPUT_DIR}" ]; then
    echo "Raw DPG image folder not found: ${RAW_OUTPUT_DIR}" >&2
    exit 1
fi

USE_REG=0
if [ -n "${MODEL_PATH}" ] && [[ "${MODEL_PATH}" =~ variant[0-9]+_2(/|$) ]]; then
    USE_REG=1
    echo "Detected variant_2 model, enabling --reg."
fi

echo "========================================="
echo "DPG pipeline"
echo "========================================="
if [ -n "${MODEL_PATH}" ]; then
    echo "Model path: ${MODEL_PATH}"
fi
if [ -n "${METADATA_FILE}" ]; then
    echo "Metadata file: ${METADATA_FILE}"
fi
if [ "${SKIP_GRID}" = "1" ]; then
    echo "Grid image dir: ${SCORE_IMAGE_ROOT}"
else
    echo "Raw output dir: ${RAW_OUTPUT_DIR}"
    echo "Grid output dir: ${GRID_OUTPUT_DIR}"
fi
echo "Score image dir: ${SCORE_IMAGE_ROOT}"
echo "DPG CSV: ${DPG_CSV_PATH}"
echo "DPG VQA model path/id: ${DPG_SCORE_VQA_MODEL_PATH}"
echo "DPG grid workers: ${DPG_GRID_WORKERS}"
echo "GPUs: ${GPUS}"
echo

if [ "${SKIP_GENERATION}" != "1" ]; then
    mkdir -p "${RAW_OUTPUT_DIR}"
    echo "Step 1: Generating DPG images..."
    if [ "${USE_REG}" = "1" ]; then
        torchrun \
            --nnodes=1 \
            --node_rank=0 \
            --nproc_per_node="${GPUS}" \
            --master_addr=127.0.0.1 \
            --master_port="${MASTER_PORT}" \
            ./eval/gen/gen_images_mp_dllm.py \
            --output_dir "${RAW_OUTPUT_DIR}" \
            --metadata_file "${METADATA_FILE}" \
            --batch_size "${BATCH_SIZE}" \
            --num_images "${NUM_IMAGES}" \
            --resolution "${RESOLUTION}" \
            --max_latent_size "${MAX_LATENT_SIZE}" \
            --model-path "${MODEL_PATH}" \
            --dpg_bench \
            --reg
    else
        torchrun \
            --nnodes=1 \
            --node_rank=0 \
            --nproc_per_node="${GPUS}" \
            --master_addr=127.0.0.1 \
            --master_port="${MASTER_PORT}" \
            ./eval/gen/gen_images_mp_dllm.py \
            --output_dir "${RAW_OUTPUT_DIR}" \
            --metadata_file "${METADATA_FILE}" \
            --batch_size "${BATCH_SIZE}" \
            --num_images "${NUM_IMAGES}" \
            --resolution "${RESOLUTION}" \
            --max_latent_size "${MAX_LATENT_SIZE}" \
            --model-path "${MODEL_PATH}" \
            --dpg_bench
    fi
fi

if [ "${SKIP_GRID}" != "1" ]; then
    echo "Step 2: Building 2x2 DPG grids..."
    GRID_BUILD_CMD=(
        python ./eval/gen/dpg_bench/build_dpg_grids.py
        --input-root "${RAW_OUTPUT_DIR}"
        --output-root "${GRID_OUTPUT_DIR}"
        --resolution "${RESOLUTION}"
        --pic-num "${DPG_SCORE_PIC_NUM}"
        --num-workers "${DPG_GRID_WORKERS}"
        --output-format "${DPG_GRID_FORMAT}"
        --png-compress-level "${DPG_GRID_PNG_COMPRESS_LEVEL}"
        --jpeg-quality "${DPG_GRID_JPEG_QUALITY}"
    )

    if [ "${DPG_GRID_OVERWRITE}" = "1" ]; then
        GRID_BUILD_CMD+=(--overwrite)
    fi

    "${GRID_BUILD_CMD[@]}"
fi

if [ "${SKIP_SCORE}" != "1" ]; then
    echo "Step 3: Calculating DPG score..."
    echo "Scoring images from: ${SCORE_IMAGE_ROOT}"
    ACCELERATE_CMD=(
        accelerate launch
        --num_machines 1
        --num_processes "${DPG_SCORE_PROCESSES}"
        --mixed_precision fp16
        --main_process_port "${DPG_SCORE_PORT}"
    )

    if [ "${DPG_SCORE_PROCESSES}" -gt 1 ]; then
        ACCELERATE_CMD+=(--multi_gpu)
    fi

    ACCELERATE_CMD+=(
        ./eval/gen/dpg_bench/compute_dpg_bench.py
        --image-root-path "${SCORE_IMAGE_ROOT}"
        --resolution "${RESOLUTION}"
        --csv "${DPG_CSV_PATH}"
        --res-path "${DPG_SCORE_RESULT_PATH}"
        --pic-num "${DPG_SCORE_PIC_NUM}"
        --vqa-model "${DPG_SCORE_VQA_MODEL}"
        --vqa-model-path "${DPG_SCORE_VQA_MODEL_PATH}"
    )

    "${ACCELERATE_CMD[@]}"
fi

echo
echo "Done."
if [ "${SKIP_GRID}" != "1" ]; then
    echo "Raw images: ${RAW_OUTPUT_DIR}"
fi
if [ "${SKIP_GRID}" != "1" ]; then
    echo "Grid images: ${GRID_OUTPUT_DIR}"
fi
if [ "${SKIP_SCORE}" != "1" ]; then
    echo "Scored images: ${SCORE_IMAGE_ROOT}"
    echo "DPG score file: ${DPG_SCORE_RESULT_PATH}"
fi
