#!/usr/bin/env bash
# Copyright 2026 LLaDA-o contributors.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=gui-ocr-align
#SBATCH --account=a0201
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
ENVIRONMENT_FILE="${ENVIRONMENT_FILE:-${REPO_ROOT}/lladao.toml}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${SCRATCH}/datasets/lladao_gui_benchmarks}"
WORK_DIR="${WORK_DIR:-${SCRATCH}/datasets/lladao_gui_benchmarks_ocr_work}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH}/datasets/lladao_gui_benchmarks_ocr}"
EASYOCR_MODEL_DIR="${EASYOCR_MODEL_DIR:-${SCRATCH}/models/easyocr}"
OCR_VENV="${OCR_VENV:-${SCRATCH}/venvs/lladao-easyocr-1.7.2}"

cd "${REPO_ROOT}"
for required in \
  "${ENVIRONMENT_FILE}" \
  "${BENCHMARK_ROOT}/manifest.json" \
  "${REPO_ROOT}/scripts/data/realign_gui_grounding_ocr.py"; do
  [[ -f "${required}" ]] || {
    echo "error: required input does not exist: ${required}" >&2
    exit 1
  }
done
mkdir -p "${WORK_DIR}" "${EASYOCR_MODEL_DIR}"

export BENCHMARK_ROOT WORK_DIR OUTPUT_ROOT EASYOCR_MODEL_DIR OCR_VENV

# Install the pinned OCR wrapper in a separate system-site-packages venv and
# download its two English model files once before launching sharded workers.
# Keeping OCR dependencies separate avoids mutating a live training process's
# environment.
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
    python -m pip install --upgrade-strategy only-if-needed "easyocr==1.7.2"
    python -c "import easyocr, os; easyocr.Reader(['\''en'\''], gpu=False, model_storage_directory=os.environ['\''EASYOCR_MODEL_DIR'\''], download_enabled=True)"
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
    exec python "${LLADAO_REPO_ROOT}/scripts/data/realign_gui_grounding_ocr.py" detect \
      --benchmark-root "${BENCHMARK_ROOT}" \
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
    exec python "${LLADAO_REPO_ROOT}/scripts/data/realign_gui_grounding_ocr.py" finalize \
      --benchmark-root "${BENCHMARK_ROOT}" \
      --work-dir "${WORK_DIR}" \
      --output-root "${OUTPUT_ROOT}" \
      --force
  '

echo "OCR-aligned benchmark: ${OUTPUT_ROOT}"
