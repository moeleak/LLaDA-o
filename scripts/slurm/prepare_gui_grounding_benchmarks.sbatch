#!/usr/bin/env bash
# Copyright 2026 LLaDA-o contributors.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=gui-bench-data
#SBATCH --account=a0201
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
ENVIRONMENT_FILE="${ENVIRONMENT_FILE:-${REPO_ROOT}/lladao.toml}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${SCRATCH}/datasets/lladao_gui_benchmarks}"
MIND2WEB_CROP_SIZE="${MIND2WEB_CROP_SIZE:-1280}"

cd "${REPO_ROOT}"
[[ -f "${ENVIRONMENT_FILE}" ]] || {
  echo "error: EDF file does not exist: ${ENVIRONMENT_FILE}" >&2
  exit 1
}

export BENCHMARK_ROOT MIND2WEB_CROP_SIZE
srun \
  --environment="${ENVIRONMENT_FILE}" \
  --network=disable_rdzv_get \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-task=1 \
  --kill-on-bad-exit=1 \
  bash -c '
    set -euo pipefail
    source "${LLADAO_REPO_ROOT}/scripts/bootstrap_lladao_env.sh"
    args=(
      all
      --root "${BENCHMARK_ROOT}"
      --mind2web-crop-size "${MIND2WEB_CROP_SIZE}"
    )
    if [[ "${FORCE_REBUILD:-0}" == "1" ]]; then
      args+=(--force)
    fi
    if [[ -n "${VISUALWEBARENA_JSONL:-}" ]]; then
      args+=(--visualwebarena-jsonl "${VISUALWEBARENA_JSONL}")
    fi
    exec python "${LLADAO_REPO_ROOT}/scripts/data/prepare_gui_grounding_benchmarks.py" "${args[@]}"
  '
