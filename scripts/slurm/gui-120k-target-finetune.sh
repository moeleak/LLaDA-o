#!/usr/bin/env bash
# Copyright 2026 LLaDA-o contributors.
# SPDX-License-Identifier: Apache-2.0

# Continue the existing GUI checkpoint on the paper-sized 120K mixture after
# correcting Mind2Web to target-explicit prompts and its published 20K bucket.
# This is the production alignment run; the separate Mind2Web-only launcher is
# only a Table 3 ablation.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"

: "${SCRATCH:?SCRATCH must be set by the Clariden login environment}"

export TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-${SCRATCH}/datasets/lladao_gui_120k_target/parquet}"
export TRAIN_RESULTS_DIR="${TRAIN_RESULTS_DIR:-${SCRATCH}/runs/lladao_gui_120k_target}"
export RESULTS_DIR="${TRAIN_RESULTS_DIR}"
export RESUME_FROM="${RESUME_FROM:-${SCRATCH}/runs/lladao_gui_120k/checkpoints/0010000}"
export JOB_NAME="${JOB_NAME:-gui-120k-target}"
export WANDB_NAME="${WANDB_NAME:-gui-120k-target-grounding}"

# Table 4 does not publish epochs or optimizer settings.  At the observed
# ~16.97 global samples/step, 7,500 steps are about 1.06 passes over 120K.
# Checkpoints every 500 steps allow evaluation-driven selection without
# pretending that the unpublished author schedule is known.
export TOTAL_STEPS="${TOTAL_STEPS:-7501}"
export SAVE_EVERY="${SAVE_EVERY:-500}"
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
