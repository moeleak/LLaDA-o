#!/usr/bin/env bash
# Copyright 2026 LLaDA-o contributors.
# SPDX-License-Identifier: Apache-2.0

# Continue a target-explicit checkpoint on the paper-aligned 120K corpus after
# replacing eligible Mind2Web DOM targets with linked OCR text boxes.  This is
# kept in a separate results directory so DOM- and OCR-annotation runs remain
# independently auditable.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"

: "${SCRATCH:?SCRATCH must be set by the Clariden login environment}"

export TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-${SCRATCH}/datasets/lladao_gui_120k_target_ocr/parquet}"
export TRAIN_RESULTS_DIR="${TRAIN_RESULTS_DIR:-${SCRATCH}/runs/lladao_gui_120k_target_ocr}"
export RESULTS_DIR="${TRAIN_RESULTS_DIR}"
export RESUME_FROM="${RESUME_FROM:-${SCRATCH}/runs/lladao_gui_120k_target/checkpoints/0001500}"
export JOB_NAME="${JOB_NAME:-gui-120k-ocr}"
export WANDB_NAME="${WANDB_NAME:-gui-120k-ocr-grounding}"

# The paper does not publish its 120K optimizer schedule.  Save densely during
# this annotation-adaptation stage and select checkpoints only from held-out
# Mind2Web plus ScreenSpot results.
export TOTAL_STEPS="${TOTAL_STEPS:-4001}"
export SAVE_EVERY="${SAVE_EVERY:-250}"
export LOG_EVERY="${LOG_EVERY:-10}"
export WARMUP_STEPS="${WARMUP_STEPS:-100}"
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

for required in \
  "${TRAIN_DATA_DIR}/manifest.json" \
  "${RESUME_FROM}/ema.safetensors" \
  "${REPO_ROOT}/scripts/slurm/gui-120k-grounding-finetune.sh"; do
  [[ -e "${required}" ]] || {
    echo "error: required input does not exist: ${required}" >&2
    exit 1
  }
done

exec bash "${REPO_ROOT}/scripts/slurm/gui-120k-grounding-finetune.sh" "$@"
