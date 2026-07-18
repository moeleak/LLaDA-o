#!/usr/bin/env bash
# Copyright 2026 LLaDA-o contributors.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=gui-benchmark
#SBATCH --account=a0201
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
ENVIRONMENT_FILE="${ENVIRONMENT_FILE:-${REPO_ROOT}/lladao.toml}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${SCRATCH}/datasets/lladao_gui_benchmarks}"
MODEL_PATH="${MODEL_PATH:-${SCRATCH}/models/GSAI-ML-LLaDA-o}"
CHECKPOINT="${CHECKPOINT:-${SCRATCH}/runs/lladao_gui_120k/checkpoints/0010000/ema.safetensors}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/runs/lladao_gui_benchmark/step-0010000/s64-b64-ct095}"
BENCHMARKS="${BENCHMARKS:-mind2web,screenspot_web_text,screenspot_web_icon,visualwebarena}"
BLOCK_LENGTH="${BLOCK_LENGTH:-64}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-64}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.95}"
WARMUP="${WARMUP:-1}"

cd "${REPO_ROOT}"
for required in \
  "${ENVIRONMENT_FILE}" \
  "${BENCHMARK_ROOT}/manifest.json" \
  "${MODEL_PATH}/llm_config.json" \
  "${CHECKPOINT}"; do
  [[ -f "${required}" ]] || {
    echo "error: required file does not exist: ${required}" >&2
    exit 1
  }
done
mkdir -p "${OUTPUT_DIR}"

export \
  BENCHMARK_ROOT \
  MODEL_PATH \
  CHECKPOINT \
  OUTPUT_DIR \
  BENCHMARKS \
  BLOCK_LENGTH \
  DIFFUSION_STEPS \
  CONFIDENCE_THRESHOLD \
  WARMUP

echo "Benchmark root: ${BENCHMARK_ROOT}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUTPUT_DIR}"
echo "Workers: ${SLURM_NTASKS}"

srun \
  --environment="${ENVIRONMENT_FILE}" \
  --network=disable_rdzv_get \
  --nodes="${SLURM_NNODES}" \
  --ntasks="${SLURM_NTASKS}" \
  --ntasks-per-node="${SLURM_NTASKS}" \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  --kill-on-bad-exit=1 \
  bash -c '
    set -euo pipefail
    source "${LLADAO_REPO_ROOT}/scripts/bootstrap_lladao_env.sh"
    args=(
      --benchmark-root "${BENCHMARK_ROOT}"
      --output-dir "${OUTPUT_DIR}"
      --model-path "${MODEL_PATH}"
      --checkpoint "${CHECKPOINT}"
      --benchmarks "${BENCHMARKS}"
      --block-length "${BLOCK_LENGTH}"
      --diffusion-steps "${DIFFUSION_STEPS}"
      --confidence-threshold "${CONFIDENCE_THRESHOLD}"
      --warmup "${WARMUP}"
    )
    if [[ -n "${EVAL_LIMIT:-}" ]]; then
      args+=(--limit "${EVAL_LIMIT}")
    fi
    exec python "${LLADAO_REPO_ROOT}/eval/gui_grounding/run_benchmark.py" "${args[@]}"
  '

score_args=(
  --benchmark-root "${BENCHMARK_ROOT}"
  --predictions-dir "${OUTPUT_DIR}"
  --output-dir "${OUTPUT_DIR}/scores"
  --benchmarks "${BENCHMARKS}"
)
if [[ -n "${EVAL_LIMIT:-}" ]]; then
  score_args+=(--limit "${EVAL_LIMIT}")
fi

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
    exec python "${LLADAO_REPO_ROOT}/eval/gui_grounding/score_benchmark.py" "$@"
  ' bash "${score_args[@]}"

echo "Results: ${OUTPUT_DIR}/scores/results.json"
echo "Table:   ${OUTPUT_DIR}/scores/results.csv"
