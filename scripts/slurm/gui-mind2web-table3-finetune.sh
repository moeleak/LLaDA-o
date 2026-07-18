#!/usr/bin/env bash
# Copyright 2026 LLaDA-o contributors.
# SPDX-License-Identifier: Apache-2.0

# Submit the paper-style Mind2Web-only ablation corresponding to Table 3 of
# arXiv:2603.26211.  The paper publishes ten epochs and linear masking, but not
# its optimizer, learning rate, batch size, prompt template, or OCR code.  The
# defaults below retain this repository's optimizer recipe and save densely
# around the estimated ten-epoch point so the logged global sample count can
# select the closest checkpoint after training.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"

: "${SCRATCH:?SCRATCH must be set by the Clariden login environment}"

export TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-${SCRATCH}/datasets/lladao_gui_mind2web_target/parquet/mind2web}"
export TRAIN_RESULTS_DIR="${TRAIN_RESULTS_DIR:-${SCRATCH}/runs/lladao_gui_mind2web_10ep}"
export RESULTS_DIR="${TRAIN_RESULTS_DIR}"
export JOB_NAME="${JOB_NAME:-gui-m2w-table3}"
export WANDB_NAME="${WANDB_NAME:-mind2web-target-grounding-10ep}"

# The prior eight-GPU run packed about 16.97 samples per optimizer step.
# 7,341 usable rows * 10 / 16.97 = 4,326 estimated steps.  Save every 250
# steps and continue through 4,750 so checkpoints bracket the true ten-epoch
# point even if Mind2Web's packing density differs from the mixed corpus.
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

for required in "${TRAIN_DATA_DIR}" "${REPO_ROOT}/scripts/slurm/gui-120k-grounding-finetune.sh"; do
  [[ -e "${required}" ]] || {
    echo "error: required input does not exist: ${required}" >&2
    exit 1
  }
done

exec bash "${REPO_ROOT}/scripts/slurm/gui-120k-grounding-finetune.sh" "$@"
