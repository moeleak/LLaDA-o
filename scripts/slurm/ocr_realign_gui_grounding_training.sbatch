#!/usr/bin/env bash
# Copyright 2026 LLaDA-o contributors.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=gui-train-ocr
#SBATCH --account=a0201
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
ENVIRONMENT_FILE="${ENVIRONMENT_FILE:-${REPO_ROOT}/lladao.toml}"
INPUT_ROOT="${INPUT_ROOT:-${SCRATCH}/datasets/lladao_gui_120k_target/parquet}"
WORK_DIR="${WORK_DIR:-${SCRATCH}/datasets/lladao_gui_120k_target_ocr_work}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH}/datasets/lladao_gui_120k_target_ocr/parquet}"
EASYOCR_MODEL_DIR="${EASYOCR_MODEL_DIR:-${SCRATCH}/models/easyocr}"
OCR_VENV="${OCR_VENV:-${SCRATCH}/venvs/lladao-easyocr-1.7.2}"

cd "${REPO_ROOT}"
for required in \
  "${ENVIRONMENT_FILE}" \
  "${INPUT_ROOT}/manifest.json" \
  "${REPO_ROOT}/scripts/data/realign_gui_grounding_training_ocr.py"; do
  [[ -f "${required}" ]] || {
    echo "error: required input does not exist: ${required}" >&2
    exit 1
  }
done
mkdir -p "${WORK_DIR}" "${EASYOCR_MODEL_DIR}"
export INPUT_ROOT WORK_DIR OUTPUT_ROOT EASYOCR_MODEL_DIR OCR_VENV

srun \
  --environment="${ENVIRONMENT_FILE}" \
  --network=disable_rdzv_get \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-task=1 \
  bash -c '
    set -euo pipefail
    source "${LLADAO_REPO_ROOT}/scripts/bootstrap_lladao_env.sh"
    if [[ ! -x "${OCR_VENV}/bin/python" ]]; then
      python -m venv --system-site-packages "${OCR_VENV}"
    fi
    source "${OCR_VENV}/bin/activate"
    python -m pip install --upgrade --upgrade-strategy only-if-needed \
      "numpy==1.26.4" \
      "opencv-python-headless==4.11.0.86" \
      "pyarrow==17.0.0" \
      "easyocr==1.7.2"
  '

srun \
  --environment="${ENVIRONMENT_FILE}" \
  --network=disable_rdzv_get \
  --nodes=1 \
  --ntasks="${SLURM_NTASKS}" \
  --ntasks-per-node="${SLURM_NTASKS}" \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  --kill-on-bad-exit=1 \
  bash -c '
    set -euo pipefail
    source "${LLADAO_REPO_ROOT}/scripts/bootstrap_lladao_env.sh"
    source "${OCR_VENV}/bin/activate"
    exec python "${LLADAO_REPO_ROOT}/scripts/data/realign_gui_grounding_training_ocr.py" rewrite \
      --input-root "${INPUT_ROOT}" \
      --work-dir "${WORK_DIR}" \
      --model-dir "${EASYOCR_MODEL_DIR}" \
      --rank "${SLURM_PROCID}" \
      --world-size "${SLURM_NTASKS}" \
      --no-download
  '

srun \
  --environment="${ENVIRONMENT_FILE}" \
  --network=disable_rdzv_get \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-task=1 \
  bash -c '
    set -euo pipefail
    source "${LLADAO_REPO_ROOT}/scripts/bootstrap_lladao_env.sh"
    python "${LLADAO_REPO_ROOT}/scripts/data/realign_gui_grounding_training_ocr.py" finalize \
      --input-root "${INPUT_ROOT}" \
      --work-dir "${WORK_DIR}" \
      --output-root "${OUTPUT_ROOT}" \
      --force
    exec python "${LLADAO_REPO_ROOT}/scripts/data/prepare_gui_grounding.py" validate \
      --root "$(dirname "${OUTPUT_ROOT}")" \
      --deep
  '

echo "OCR-aligned 120K training corpus: ${OUTPUT_ROOT}"
