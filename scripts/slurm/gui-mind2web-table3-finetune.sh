#!/usr/bin/env bash
# Copyright 2026 LLaDA-o contributors.
# SPDX-License-Identifier: Apache-2.0

# Submit the highlighted Mind2Web-only ablation corresponding to Table 3 of
# arXiv:2603.26211: 7,341 unique target-preserving crops, OCR-linked target
# annotations, ten epochs, and linear masking.  The paper does not publish its
# optimizer, learning rate, batch size, prompt template, crop seed, or OCR
# matcher.  The defaults below retain this repository's optimizer recipe and
# save densely around the measured ten-epoch point so held-out evaluation can
# select the best checkpoint without using the test score for training.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"

: "${SCRATCH:?SCRATCH must be set by the Clariden login environment}"

export TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-${SCRATCH}/datasets/lladao_gui_mind2web_target_ocr/parquet/mind2web}"
export TRAIN_RESULTS_DIR="${TRAIN_RESULTS_DIR:-${SCRATCH}/runs/lladao_gui_mind2web_table3_ocr_10ep}"
export RESULTS_DIR="${TRAIN_RESULTS_DIR}"
export RESUME_FROM="${RESUME_FROM:-${SCRATCH}/models/GSAI-ML-LLaDA-o}"
export JOB_NAME="${JOB_NAME:-gui-m2w-t3-ocr}"
export WANDB_NAME="${WANDB_NAME:-mind2web-table3-ocr-10ep}"

# The measured eight-GPU Mind2Web-only run consumed about 16 samples per
# optimizer step.  7,341 rows * 10 / 16 = 4,588 estimated steps.  Saving every
# 250 steps and continuing through 4,750 brackets ten epochs despite small
# packing-density changes.
export TOTAL_STEPS="${TOTAL_STEPS:-4751}"
export SAVE_EVERY="${SAVE_EVERY:-250}"
export LOG_EVERY="${LOG_EVERY:-10}"
export WARMUP_STEPS="${WARMUP_STEPS:-300}"
export LEARNING_RATE="${LEARNING_RATE:-2.5e-5}"
export LR_SCHEDULER="${LR_SCHEDULER:-constant}"
export EMA_DECAY="${EMA_DECAY:-0.995}"
export EXPECTED_NUM_TOKENS="${EXPECTED_NUM_TOKENS:-6144}"
export MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-8192}"
export MAX_NUM_TOKENS_PER_SAMPLE="${MAX_NUM_TOKENS_PER_SAMPLE:-8192}"
export NNODES="${NNODES:-2}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export WALLTIME="${WALLTIME:-12:00:00}"
export CHAIN_JOBS="${CHAIN_JOBS:-2}"

for required in "${REPO_ROOT}/scripts/slurm/gui-120k-grounding-finetune.sh"; do
  [[ -e "${required}" ]] || {
    echo "error: required input does not exist: ${required}" >&2
    exit 1
  }
done
if [[ ! -d "${TRAIN_DATA_DIR}" ]]; then
  if [[ -z "${AFTER_JOB_ID:-}" ]]; then
    echo "error: training data does not exist: ${TRAIN_DATA_DIR}" >&2
    exit 1
  fi
  echo "Training data will be validated after dependency job ${AFTER_JOB_ID}: ${TRAIN_DATA_DIR}"
fi
if [[ ! -f "${RESUME_FROM}/ema.safetensors" && ! -f "${RESUME_FROM}/ema.safetensors.index.json" ]]; then
  echo "error: base checkpoint is missing below ${RESUME_FROM}" >&2
  exit 1
fi

exec bash "${REPO_ROOT}/scripts/slurm/gui-120k-grounding-finetune.sh" "$@"
