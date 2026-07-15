#!/usr/bin/env bash
# Copyright 2026 LLaDA-o contributors.
# SPDX-License-Identifier: Apache-2.0

# Submit the one-node Clariden GUI grounding fine-tuning job from a login node.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
BATCH_SCRIPT="${REPO_ROOT}/scripts/slurm/train_gui_grounding_120k.sbatch"
ENVIRONMENT_FILE="${ENVIRONMENT_FILE:-${REPO_ROOT}/lladao.toml}"

: "${SCRATCH:?SCRATCH must be set by the Clariden login environment}"

ACCOUNT="${ACCOUNT:-a0201}"
PARTITION="${PARTITION:-normal}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEMORY="${MEMORY:-450G}"
WALLTIME="${WALLTIME:-12:00:00}"

export REPO_ROOT
export ENVIRONMENT_FILE
export RESULTS_DIR="${RESULTS_DIR:-${SCRATCH}/runs/lladao_gui_120k}"
export TOTAL_STEPS="${TOTAL_STEPS:-10001}"
export SAVE_EVERY="${SAVE_EVERY:-500}"
export LOG_EVERY="${LOG_EVERY:-10}"
export WANDB_NAME="${WANDB_NAME:-gui-grounding-1node}"
export EXPECTED_NUM_TOKENS="${EXPECTED_NUM_TOKENS:-8192}"
export MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-12288}"
export GPUS_PER_NODE

[[ -f "${BATCH_SCRIPT}" ]] || {
  echo "error: batch script does not exist: ${BATCH_SCRIPT}" >&2
  exit 1
}
[[ -f "${ENVIRONMENT_FILE}" ]] || {
  echo "error: EDF file does not exist: ${ENVIRONMENT_FILE}" >&2
  exit 1
}

cd "${REPO_ROOT}"

submit_command=(
  sbatch
  --parsable
  -A "${ACCOUNT}"
  -p "${PARTITION}"
  --nodes=1
  --ntasks-per-node=1
  --gres="gpu:${GPUS_PER_NODE}"
  --cpus-per-task="${CPUS_PER_TASK}"
  --exclusive
  --mem="${MEMORY}"
  --time="${WALLTIME}"
  "$@"
  --export=ALL
  "${BATCH_SCRIPT}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'Environment:'
  printf ' %s=%q' \
    RESULTS_DIR "${RESULTS_DIR}" \
    TOTAL_STEPS "${TOTAL_STEPS}" \
    SAVE_EVERY "${SAVE_EVERY}" \
    EXPECTED_NUM_TOKENS "${EXPECTED_NUM_TOKENS}" \
    MAX_NUM_TOKENS "${MAX_NUM_TOKENS}"
  printf '\nCommand:'
  printf ' %q' "${submit_command[@]}"
  printf '\n'
  exit 0
fi

job_spec="$("${submit_command[@]}")"
job_id="${job_spec%%;*}"
log_path="${REPO_ROOT}/slurm-lladao-gui-120k-${job_id}.out"

printf 'Submitted job %s\n' "${job_id}"
printf 'Results: %s\n' "${RESULTS_DIR}"
printf 'Log: %s\n' "${log_path}"
printf 'Monitor: squeue -j %q\n' "${job_id}"
printf 'Follow: tail -F %q\n' "${log_path}"
