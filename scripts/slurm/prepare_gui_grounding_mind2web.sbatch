#!/usr/bin/env bash
# Copyright 2026 LLaDA-o contributors.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=gui-m2w-data
#SBATCH --account=a0201
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
ENVIRONMENT_FILE="${ENVIRONMENT_FILE:-${REPO_ROOT}/lladao.toml}"
SOURCE_ROOT="${SOURCE_ROOT:-${SCRATCH}/datasets/lladao_gui_120k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH}/datasets/lladao_gui_mind2web_target}"
MIND2WEB_CROP_SIZE="${MIND2WEB_CROP_SIZE:-1280}"

cd "${REPO_ROOT}"
for required in "${ENVIRONMENT_FILE}" "${SOURCE_ROOT}/raw/mind2web"; do
  [[ -e "${required}" ]] || {
    echo "error: required input does not exist: ${required}" >&2
    exit 1
  }
done

mkdir -p "${OUTPUT_ROOT}"
if [[ ! -e "${OUTPUT_ROOT}/raw" ]]; then
  ln -s "${SOURCE_ROOT}/raw" "${OUTPUT_ROOT}/raw"
fi
[[ "$(readlink -f "${OUTPUT_ROOT}/raw")" == "$(readlink -f "${SOURCE_ROOT}/raw")" ]] || {
  echo "error: ${OUTPUT_ROOT}/raw does not point to ${SOURCE_ROOT}/raw" >&2
  exit 1
}

export OUTPUT_ROOT MIND2WEB_CROP_SIZE
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
    python "${LLADAO_REPO_ROOT}/scripts/data/prepare_gui_grounding.py" \
      build \
      --root "${OUTPUT_ROOT}" \
      --sources mind2web \
      --mind2web-count 7341 \
      --mind2web-crop-size "${MIND2WEB_CROP_SIZE}" \
      --mind2web-crop-mode random \
      --mind2web-prompt-protocol target_grounding \
      --force
    exec python "${LLADAO_REPO_ROOT}/scripts/data/prepare_gui_grounding.py" \
      validate \
      --root "${OUTPUT_ROOT}" \
      --deep
  '
