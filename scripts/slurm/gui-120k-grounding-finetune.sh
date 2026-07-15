#!/usr/bin/env bash
# Copyright 2026 LLaDA-o contributors.
# SPDX-License-Identifier: Apache-2.0

# Submit the two-node, eight-GPU Clariden GUI grounding fine-tuning job from a
# login node. Clariden GH200 nodes expose four GPUs each.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
BATCH_SCRIPT="${REPO_ROOT}/scripts/slurm/train_gui_grounding_120k.sbatch"
ENVIRONMENT_FILE="${ENVIRONMENT_FILE:-${REPO_ROOT}/lladao.toml}"

: "${SCRATCH:?SCRATCH must be set by the Clariden login environment}"

ACCOUNT="${ACCOUNT:-a0201}"
PARTITION="${PARTITION:-normal}"
NNODES="${NNODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEMORY="${MEMORY:-450G}"
WALLTIME="${WALLTIME:-12:00:00}"
JOB_NAME="${JOB_NAME:-gui-grounding}"
CHAIN_JOBS="${CHAIN_JOBS:-1}"
AFTER_JOB_ID="${AFTER_JOB_ID:-}"

export REPO_ROOT
export ENVIRONMENT_FILE
export RESULTS_DIR="${RESULTS_DIR:-${SCRATCH}/runs/lladao_gui_120k}"
export TOTAL_STEPS="${TOTAL_STEPS:-10001}"
export SAVE_EVERY="${SAVE_EVERY:-500}"
export LOG_EVERY="${LOG_EVERY:-10}"
export WANDB_NAME="${WANDB_NAME:-gui-grounding}"
export EXPECTED_NUM_TOKENS="${EXPECTED_NUM_TOKENS:-6144}"
export MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-8192}"
export MAX_NUM_TOKENS_PER_SAMPLE="${MAX_NUM_TOKENS_PER_SAMPLE:-8192}"
export NNODES
export GPUS_PER_NODE

[[ "${NNODES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: NNODES must be a positive integer (got ${NNODES})" >&2
  exit 1
}
[[ "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: GPUS_PER_NODE must be a positive integer (got ${GPUS_PER_NODE})" >&2
  exit 1
}
[[ "${CHAIN_JOBS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: CHAIN_JOBS must be a positive integer (got ${CHAIN_JOBS})" >&2
  exit 1
}
if [[ -n "${AFTER_JOB_ID}" && ! "${AFTER_JOB_ID}" =~ ^[0-9]+(_[0-9]+)?$ ]]; then
  echo "error: AFTER_JOB_ID must be a Slurm job ID (got ${AFTER_JOB_ID})" >&2
  exit 1
fi

[[ -f "${BATCH_SCRIPT}" ]] || {
  echo "error: batch script does not exist: ${BATCH_SCRIPT}" >&2
  exit 1
}
[[ -f "${ENVIRONMENT_FILE}" ]] || {
  echo "error: EDF file does not exist: ${ENVIRONMENT_FILE}" >&2
  exit 1
}

cd "${REPO_ROOT}"

extra_sbatch_args=("$@")

build_submit_command() {
  local dependency_job_id="$1"

  submit_command=(
    sbatch
    --parsable
    -A "${ACCOUNT}"
    -p "${PARTITION}"
    --job-name="${JOB_NAME}"
    --nodes="${NNODES}"
    --ntasks-per-node=1
    --gres="gpu:${GPUS_PER_NODE}"
    --cpus-per-task="${CPUS_PER_TASK}"
    --exclusive
    --mem="${MEMORY}"
    --time="${WALLTIME}"
  )
  if [[ -n "${dependency_job_id}" ]]; then
    submit_command+=(--dependency="afterany:${dependency_job_id}")
  fi
  submit_command+=("${extra_sbatch_args[@]}" --export=ALL "${BATCH_SCRIPT}")
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'Environment:'
  printf ' %s=%q' \
    NNODES "${NNODES}" \
    GPUS_PER_NODE "${GPUS_PER_NODE}" \
    RESULTS_DIR "${RESULTS_DIR}" \
    TOTAL_STEPS "${TOTAL_STEPS}" \
    SAVE_EVERY "${SAVE_EVERY}" \
    EXPECTED_NUM_TOKENS "${EXPECTED_NUM_TOKENS}" \
    MAX_NUM_TOKENS "${MAX_NUM_TOKENS}" \
    MAX_NUM_TOKENS_PER_SAMPLE "${MAX_NUM_TOKENS_PER_SAMPLE}"
  printf ' CHAIN_JOBS=%q AFTER_JOB_ID=%q\n' "${CHAIN_JOBS}" "${AFTER_JOB_ID}"

  dependency_job_id="${AFTER_JOB_ID}"
  for ((chain_index = 1; chain_index <= CHAIN_JOBS; chain_index++)); do
    build_submit_command "${dependency_job_id}"
    printf 'Command %d:' "${chain_index}"
    printf ' %q' "${submit_command[@]}"
    printf '\n'
    dependency_job_id="<job-id-from-command-${chain_index}>"
  done
  exit 0
fi

submitted_job_ids=()
dependency_job_id="${AFTER_JOB_ID}"
for ((chain_index = 1; chain_index <= CHAIN_JOBS; chain_index++)); do
  build_submit_command "${dependency_job_id}"
  job_spec="$("${submit_command[@]}")"
  job_id="${job_spec%%;*}"
  submitted_job_ids+=("${job_id}")
  log_path="${REPO_ROOT}/slurm-${JOB_NAME}-${job_id}.out"

  printf 'Submitted job %s' "${job_id}"
  if [[ -n "${dependency_job_id}" ]]; then
    printf ' after job %s' "${dependency_job_id}"
  fi
  printf '\nLog: %s\n' "${log_path}"
  dependency_job_id="${job_id}"
done

printf 'Results: %s\n' "${RESULTS_DIR}"
job_list="${submitted_job_ids[*]}"
job_list="${job_list// /,}"
printf 'Monitor: squeue -j %q\n' "${job_list}"
