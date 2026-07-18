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
# Start from the last held-out-validated DOM checkpoint.  Later DOM checkpoints
# are not automatically better and must only be selected after benchmark
# validation; RESUME_FROM remains overridable for a different validated run.
export RESUME_FROM="${RESUME_FROM:-${SCRATCH}/runs/lladao_gui_120k_target/checkpoints/0001000}"
export JOB_NAME="${JOB_NAME:-gui-120k-ocr}"
export WANDB_NAME="${WANDB_NAME:-gui-120k-ocr-grounding}"

# The paper does not publish its 120K optimizer schedule.  At the observed
# roughly 16 global samples per step, 7,500 steps cover about one 120K pass.
# Save densely and select checkpoints only from held-out Mind2Web plus
# ScreenSpot results; an earlier checkpoint can stop the chain if it reaches
# the target without harming cross-domain scores.
export TOTAL_STEPS="${TOTAL_STEPS:-7501}"
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
export CHAIN_JOBS="${CHAIN_JOBS:-3}"

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
