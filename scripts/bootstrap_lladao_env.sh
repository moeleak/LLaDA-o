#!/usr/bin/env bash
# Idempotent bootstrap sourced after Pyxis has entered the container. The
# virtual environment lives on SCRATCH and reuses the CUDA-aware PyTorch stack
# provided by the NVIDIA NGC image.

if [[ "${LLADAO_BOOTSTRAP_ACTIVE:-0}" == "1" ]]; then
    return 0 2>/dev/null || exit 0
fi
export LLADAO_BOOTSTRAP_ACTIVE=1

LLADAO_REPO_ROOT="${LLADAO_REPO_ROOT:-${SCRATCH:?SCRATCH must be set}/projects/LLaDA-o}"
LLADAO_VENV="${LLADAO_VENV:-${SCRATCH}/venvs/lladao-ngc-25.01-v2}"
export LLADAO_REPO_ROOT LLADAO_VENV

ENV_SPEC_VERSION="ngc-25.01-2026-07-15-v4"
STAMP_FILE="${LLADAO_VENV}/.lladao-env-version"
LOCK_FILE="${LLADAO_VENV}.lock"

mkdir -p \
    "$(dirname "${LLADAO_VENV}")" \
    "${PIP_CACHE_DIR:-${SCRATCH}/.cache/pip}" \
    "${HF_HOME:-${SCRATCH}/huggingface}" \
    "${TORCH_HOME:-${SCRATCH}/.cache/torch}" \
    "${WANDB_DIR:-${SCRATCH}/wandb}" \
    "${CUDA_CACHE_PATH:-/tmp/cuda-cache}" \
    "${TRITON_CACHE_DIR:-/tmp/triton-cache}" || exit 1

base_python="$(PATH="${PATH#${LLADAO_VENV}/bin:}" command -v python3 || true)"
if [[ -z "${base_python}" ]]; then
    echo "LLaDA-o bootstrap: python3 is not available in the container" >&2
    return 1 2>/dev/null || exit 1
fi

# BASH_ENV is inherited by helper shells used while Pyxis/Enroot imports an
# image. Guard against ever creating the persistent venv with a login-node
# interpreter again.
if ! "${base_python}" - <<'PY' >/dev/null 2>&1
import sys
import torch

raise SystemExit(sys.version_info < (3, 10))
PY
then
    echo "LLaDA-o bootstrap: the NGC Python/PyTorch environment is not active" >&2
    echo "LLaDA-o bootstrap: run this script inside the container" >&2
    return 1 2>/dev/null || exit 1
fi

exec 9>"${LOCK_FILE}"
flock 9

if [[ ! -x "${LLADAO_VENV}/bin/python" ]]; then
    echo "LLaDA-o bootstrap: creating ${LLADAO_VENV}"
    "${base_python}" -m venv --system-site-packages "${LLADAO_VENV}" || exit 1
fi

# shellcheck disable=SC1091
source "${LLADAO_VENV}/bin/activate" || exit 1

installed_version=""
if [[ -f "${STAMP_FILE}" ]]; then
    installed_version="$(<"${STAMP_FILE}")"
fi

if [[ "${installed_version}" != "${ENV_SPEC_VERSION}" ]]; then
    echo "LLaDA-o bootstrap: installing Python dependencies (${ENV_SPEC_VERSION})"

    if ! (
        set -euo pipefail

        # Keep the venv compatible with packages inherited from NGC 25.01
        # (notably DALI's packaging and six constraints).
        python -m pip install --upgrade \
            "pip<24.3" \
            "setuptools<71" \
            "wheel<0.46" \
            "packaging<=24.2" \
            "six==1.16.0" \
            "ninja<1.12" || exit 1
        python -m pip install --upgrade --upgrade-strategy only-if-needed \
            "accelerate>=0.34.0" \
            atorch \
            bitsandbytes \
            "datasets>=3.6,<4" \
            "decord==0.6.0; platform_machine != 'aarch64'" \
            "decord2==3.4.0; platform_machine == 'aarch64'" \
            "einops==0.8.1" \
            gradio \
            "huggingface_hub>=0.34,<1" \
            "matplotlib>=3.8" \
            "numpy==1.26.4" \
            "opencv-python-headless==4.11.0.86" \
            "Pillow>=10" \
            "pyarrow==17.0.0" \
            "PyYAML>=6.0.2" \
            "requests>=2.32.3" \
            "safetensors>=0.4.5" \
            "scipy>=1.12" \
            "sentencepiece>=0.2.0" \
            "transformers==4.56.2" \
            wandb \
            webdataset \
            "weblinx==0.3.2" \
            xlsxwriter || exit 1

        if ! python -c 'import flash_attn' >/dev/null 2>&1; then
            MAX_JOBS="${MAX_JOBS:-4}" \
                python -m pip install --no-build-isolation "flash-attn>=2.7,<3" || exit 1
        fi

        python - <<'PY' || exit 1
import accelerate
import atorch
import datasets
import decord
import einops
import flash_attn
import huggingface_hub
import numpy
import pyarrow
import torch
import transformers
import wandb
import webdataset
import weblinx
from PIL import Image
from transformers import AutoTokenizer, DINOv3ViTModel

# Importing torch alone only warns when its NumPy ABI is incompatible. Exercise
# the bridge so the bootstrap cannot stamp a broken environment as ready.
torch.from_numpy(numpy.zeros(1, dtype=numpy.float32))

print(f"LLaDA-o bootstrap: Python {__import__('sys').version.split()[0]}")
print(f"LLaDA-o bootstrap: PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
print(f"LLaDA-o bootstrap: NumPy {numpy.__version__}, PyArrow {pyarrow.__version__}")
print(f"LLaDA-o bootstrap: Transformers {transformers.__version__}")
PY

        printf '%s\n' "${ENV_SPEC_VERSION}" >"${STAMP_FILE}" || exit 1
    ); then
        echo "LLaDA-o bootstrap: dependency installation failed" >&2
        flock -u 9
        exec 9>&-
        exit 1
    fi

    echo "LLaDA-o bootstrap: environment is ready"
fi

flock -u 9
exec 9>&-
unset LLADAO_BOOTSTRAP_ACTIVE
