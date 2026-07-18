#!/usr/bin/env bash
# Copyright 2026 LLaDA-o contributors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

die() {
  echo "error: $*" >&2
  exit 1
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

: "${MODEL_PATH:?Set MODEL_PATH to the downloaded GSAI-ML/LLaDA-o checkpoint directory}"
: "${LLADAO_GUI_GROUNDING_DIR:?Set LLADAO_GUI_GROUNDING_DIR to the prepared parquet directory}"

RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results/gui_grounding_120k}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RESULTS_DIR}/checkpoints}"
WANDB_LOG_DIR="${WANDB_LOG_DIR:-${RESULTS_DIR}/wandb}"
DATASET_CONFIG_FILE="${DATASET_CONFIG_FILE:-${REPO_ROOT}/data/configs/gui_grounding_table1.yaml}"
RESUME_FROM="${RESUME_FROM:-${MODEL_PATH}}"

NNODES="${NNODES:-${SLURM_NNODES:-1}}"
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  gpu_spec="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"
  if [[ "${gpu_spec}" =~ ([0-9]+)$ ]]; then
    NPROC_PER_NODE="${BASH_REMATCH[1]}"
  else
    die "Cannot infer NPROC_PER_NODE from GPU specification: ${gpu_spec}"
  fi
fi
NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-${RANK:-0}}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29199}"
MAX_RESTARTS="${MAX_RESTARTS:-5}"

# With token-based packing, optimizer steps do not map to epochs exactly.  At
# the measured eight-GPU packing rate of roughly 16.97 global samples/step,
# 7,071 steps are one pass over the prepared 120K allocation. Treat the default
# as a starting point and verify exposure from the logged total_samples value.
# The extra final iteration makes the existing zero-based training loop save a
# checkpoint at step 10000 when SAVE_EVERY=500.
TOTAL_STEPS="${TOTAL_STEPS:-10001}"
WARMUP_STEPS="${WARMUP_STEPS:-300}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-10}"
LEARNING_RATE="${LEARNING_RATE:-2.5e-5}"
MIN_LR="${MIN_LR:-1e-7}"
LR_SCHEDULER="${LR_SCHEDULER:-constant}"
EMA_DECAY="${EMA_DECAY:-0.995}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

NUM_WORKERS="${NUM_WORKERS:-1}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
EXPECTED_NUM_TOKENS="${EXPECTED_NUM_TOKENS:-32768}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-36864}"
MAX_NUM_TOKENS_PER_SAMPLE="${MAX_NUM_TOKENS_PER_SAMPLE:-16384}"
PREFER_BUFFER_BEFORE="${PREFER_BUFFER_BEFORE:-16384}"
MAX_BUFFER_SIZE="${MAX_BUFFER_SIZE:-50}"

FREEZE_LLM="${FREEZE_LLM:-False}"
FREEZE_VIT="${FREEZE_VIT:-False}"
FREEZE_UND="${FREEZE_UND:-False}"
CPU_OFFLOAD="${CPU_OFFLOAD:-False}"
WANDB_OFFLINE="${WANDB_OFFLINE:-True}"
WANDB_PROJECT="${WANDB_PROJECT:-lladao-gui-grounding}"
WANDB_NAME="${WANDB_NAME:-gui-grounding-table1}"
WANDB_RUN_ID="${WANDB_RUN_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

for value_name in NNODES NPROC_PER_NODE TOTAL_STEPS WARMUP_STEPS SAVE_EVERY LOG_EVERY NUM_WORKERS; do
  value="${!value_name}"
  if [[ "${value_name}" == "WARMUP_STEPS" ]] && [[ "${value}" == "0" ]]; then
    continue
  fi
  is_positive_integer "${value}" || die "${value_name} must be a positive integer (got ${value})"
done

[[ -d "${MODEL_PATH}" ]] || die "MODEL_PATH does not exist: ${MODEL_PATH}"
[[ -f "${MODEL_PATH}/llm_config.json" ]] || die "Missing ${MODEL_PATH}/llm_config.json"
[[ -f "${MODEL_PATH}/vit_config.json" ]] || die "Missing ${MODEL_PATH}/vit_config.json"
if [[ ! -f "${MODEL_PATH}/ema.safetensors" && ! -f "${MODEL_PATH}/ema.safetensors.index.json" ]]; then
  die "MODEL_PATH must contain ema.safetensors or ema.safetensors.index.json"
fi
[[ -f "${DATASET_CONFIG_FILE}" ]] || die "Dataset config does not exist: ${DATASET_CONFIG_FILE}"
[[ -d "${LLADAO_GUI_GROUNDING_DIR}" ]] || die "Parquet directory does not exist: ${LLADAO_GUI_GROUNDING_DIR}"
if ! find "${LLADAO_GUI_GROUNDING_DIR}" -type f -name '*.parquet' -print -quit | grep -q .; then
  die "No parquet files found under ${LLADAO_GUI_GROUNDING_DIR}"
fi

if (( NNODES > 1 )) && [[ "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" ]]; then
  die "Multi-node training requires MASTER_ADDR to be the hostname/IP of rank 0"
fi

mkdir -p "${RESULTS_DIR}" "${CHECKPOINT_DIR}" "${WANDB_LOG_DIR}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export LLADAO_GUI_GROUNDING_DIR
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${REPO_ROOT}"

launcher=(
  "${PYTHON_BIN}" -m atorch.distributed.run
  --fault_tolerant
  --network-check
  "--max_restarts=${MAX_RESTARTS}"
  "--nnode=${NNODES}"
  "--nproc_per_node=${NPROC_PER_NODE}"
  --rdzv_conf join_timeout=5400
  "--master_addr=${MASTER_ADDR}"
  "--master_port=${MASTER_PORT}"
  "--node_rank=${NODE_RANK}"
)

training_args=(
  train/pretrain_unified_navit.py
  --dataset_config_file "${DATASET_CONFIG_FILE}"
  --model_path "${MODEL_PATH}"
  --resume_from "${RESUME_FROM}"
  --results_dir "${RESULTS_DIR}"
  --checkpoint_dir "${CHECKPOINT_DIR}"
  --wandb_log_dir "${WANDB_LOG_DIR}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_name "${WANDB_NAME}"
  --wandb_runid "${WANDB_RUN_ID}"
  --wandb_offline "${WANDB_OFFLINE}"
  --layer_module LLaDAMoTDecoderLayer
  --llm_qk_norm True
  --visual_gen False
  --visual_und True
  --visual_und_sft True
  --merge_vit_text_segments True
  --ada_len False
  --freeze_llm "${FREEZE_LLM}"
  --freeze_vit "${FREEZE_VIT}"
  --freeze_und "${FREEZE_UND}"
  --finetune_from_hf True
  --resume_model_only True
  --finetune_from_ema True
  --auto_resume True
  --sharding_strategy FULL_SHARD
  --cpu_offload "${CPU_OFFLOAD}"
  --use_flex True
  --total_steps "${TOTAL_STEPS}"
  --warmup_steps "${WARMUP_STEPS}"
  --save_every "${SAVE_EVERY}"
  --log_every "${LOG_EVERY}"
  --lr_scheduler "${LR_SCHEDULER}"
  --lr "${LEARNING_RATE}"
  --min_lr "${MIN_LR}"
  --ema "${EMA_DECAY}"
  --max_grad_norm "${MAX_GRAD_NORM}"
  --ce_weight 1.0
  --ce_loss_reweighting False
  --num_workers "${NUM_WORKERS}"
  --prefetch_factor "${PREFETCH_FACTOR}"
  --expected_num_tokens "${EXPECTED_NUM_TOKENS}"
  --max_num_tokens "${MAX_NUM_TOKENS}"
  --max_num_tokens_per_sample "${MAX_NUM_TOKENS_PER_SAMPLE}"
  --prefer_buffer_before "${PREFER_BUFFER_BEFORE}"
  --max_buffer_size "${MAX_BUFFER_SIZE}"
)

echo "Launching GUI grounding fine-tuning"
echo "  nodes / GPUs per node : ${NNODES} / ${NPROC_PER_NODE}"
echo "  node rank / master    : ${NODE_RANK} / ${MASTER_ADDR}:${MASTER_PORT}"
echo "  model                  : ${MODEL_PATH}"
echo "  data                   : ${LLADAO_GUI_GROUNDING_DIR}"
echo "  results                : ${RESULTS_DIR}"
echo "  total / save steps     : ${TOTAL_STEPS} / ${SAVE_EVERY}"
echo "  token target / maximum : ${EXPECTED_NUM_TOKENS} / ${MAX_NUM_TOKENS}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'Command:'
  printf ' %q' "${launcher[@]}" "${training_args[@]}" "$@"
  printf '\n'
  exit 0
fi

if ! dependency_error="$("${PYTHON_BIN}" -c \
  'from transformers import AutoTokenizer, DINOv3ViTModel' 2>&1)"; then
  printf '%s\n' "${dependency_error}" >&2
  die "Transformers/PyTorch compatibility check failed; use the NGC image and virtual environment from lladao.toml"
fi

visible_gpus="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
if ! is_positive_integer "${visible_gpus}" || (( visible_gpus < NPROC_PER_NODE )); then
  die "Requested ${NPROC_PER_NODE} processes but only ${visible_gpus} CUDA devices are visible"
fi

exec "${launcher[@]}" "${training_args[@]}" "$@"
