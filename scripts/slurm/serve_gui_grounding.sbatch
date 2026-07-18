#!/usr/bin/env bash
# Copyright 2026 LLaDA-o contributors.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=gui-demo
#SBATCH --account=a0201
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
ENVIRONMENT_FILE="${ENVIRONMENT_FILE:-${REPO_ROOT}/lladao.toml}"
PORT="${PORT:-7860}"
LLADAO_GUI_CHECKPOINT="${LLADAO_GUI_CHECKPOINT:-${SCRATCH}/runs/lladao_gui_120k/checkpoints/0010000/ema.safetensors}"

cd "${REPO_ROOT}"

[[ -f "${ENVIRONMENT_FILE}" ]] || {
  echo "error: EDF file does not exist: ${ENVIRONMENT_FILE}" >&2
  exit 1
}
[[ -f "${LLADAO_GUI_CHECKPOINT}" ]] || {
  echo "error: checkpoint does not exist: ${LLADAO_GUI_CHECKPOINT}" >&2
  exit 1
}

export PORT LLADAO_GUI_CHECKPOINT

echo "Starting GUI grounding demo on $(hostname):${PORT}"
echo "Checkpoint: ${LLADAO_GUI_CHECKPOINT}"

srun \
  --environment="${ENVIRONMENT_FILE}" \
  --network=disable_rdzv_get \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-task=1 \
  --kill-on-bad-exit=1 \
  bash -c 'source "${LLADAO_REPO_ROOT}/scripts/bootstrap_lladao_env.sh" && exec python "${LLADAO_REPO_ROOT}/gui_grounding_demo.py" --checkpoint "${LLADAO_GUI_CHECKPOINT}" --port "${PORT}"'
