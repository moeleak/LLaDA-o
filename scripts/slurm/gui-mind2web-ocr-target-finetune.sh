#!/usr/bin/env bash
# Copyright 2026 LLaDA-o contributors.
# SPDX-License-Identifier: Apache-2.0

# Finish a held-out-validated mixed-corpus checkpoint on OCR-aligned
# Mind2Web rows only.  This increases exposure to the paper's OCR target
# annotation without changing the benchmark protocol.  Keep the run separate
# from both the mixed OCR stage and the Table 3 from-base ablation so every
# checkpoint remains auditable.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"

: "${SCRATCH:?SCRATCH must be set by the Clariden login environment}"

export TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-${SCRATCH}/datasets/lladao_gui_120k_target_ocr/parquet/mind2web}"
export TRAIN_RESULTS_DIR="${TRAIN_RESULTS_DIR:-${SCRATCH}/runs/lladao_gui_mind2web_target_ocr}"
export RESULTS_DIR="${TRAIN_RESULTS_DIR}"
export RESUME_FROM="${RESUME_FROM:-${SCRATCH}/runs/lladao_gui_120k_target_ocr/checkpoints/0001000}"
export JOB_NAME="${JOB_NAME:-gui-m2w-ocr}"
export WANDB_NAME="${WANDB_NAME:-mind2web-ocr-grounding-finish}"

# At the measured roughly 17 global samples per optimizer step, 4,500 steps
# expose about 76K OCR-aligned rows.  This is close to the paper's published
# 7K x 10-epoch Table 3 exposure while retaining all prepared crop variants.
# The lower learning rate and dense saves make held-out early stopping the
# selection rule; TOTAL_STEPS is an upper bound, not a mandatory endpoint.
export TOTAL_STEPS="${TOTAL_STEPS:-4501}"
export SAVE_EVERY="${SAVE_EVERY:-250}"
export LOG_EVERY="${LOG_EVERY:-10}"
export WARMUP_STEPS="${WARMUP_STEPS:-50}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
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
  "${TRAIN_DATA_DIR}" \
  "${RESUME_FROM}/ema.safetensors" \
  "${REPO_ROOT}/scripts/slurm/gui-120k-grounding-finetune.sh"; do
  [[ -e "${required}" ]] || {
    echo "error: required input does not exist: ${required}" >&2
    exit 1
  }
done

exec bash "${REPO_ROOT}/scripts/slurm/gui-120k-grounding-finetune.sh" "$@"
