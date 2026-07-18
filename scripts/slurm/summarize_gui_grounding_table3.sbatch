#!/usr/bin/env bash
# Copyright 2026 LLaDA-o contributors.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=gui-t3-summary
#SBATCH --account=a0201
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
ENVIRONMENT_FILE="${ENVIRONMENT_FILE:-${REPO_ROOT}/lladao.toml}"
RESULTS_ROOT="${RESULTS_ROOT:-${SCRATCH}/runs/lladao_gui_benchmark/table3-m2w-only}"
DOM_BENCHMARK_ROOT="${DOM_BENCHMARK_ROOT:-${SCRATCH}/datasets/lladao_gui_benchmarks}"
RUN_NAME="${RUN_NAME:-s64-b64-ct095}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-475.1}"
PRIMARY_STEP="${PRIMARY_STEP:-4750}"
REQUIRED_STEPS="${REQUIRED_STEPS:-250,500,750,1000,1500,2000,2500,3000,3500,4000,4500,4750}"

for required in \
    "${ENVIRONMENT_FILE}" \
    "${DOM_BENCHMARK_ROOT}/manifest.json" \
    "${RESULTS_ROOT}"; do
    [[ -e "${required}" ]] || {
        echo "error: required path does not exist: ${required}" >&2
        exit 1
    }
done

args=(
    --results-root "${RESULTS_ROOT}"
    --run-name "${RUN_NAME}"
    --dom-benchmark-root "${DOM_BENCHMARK_ROOT}"
    --steps-per-epoch "${STEPS_PER_EPOCH}"
    --require-steps "${REQUIRED_STEPS}"
    --primary-step "${PRIMARY_STEP}"
)

echo "Results root:  ${RESULTS_ROOT}"
echo "Required:      ${REQUIRED_STEPS}"
echo "Primary step:  ${PRIMARY_STEP}"

srun \
    --environment="${ENVIRONMENT_FILE}" \
    --network=disable_rdzv_get \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --kill-on-bad-exit=1 \
    bash -c '
        set -euo pipefail
        source "${LLADAO_REPO_ROOT}/scripts/bootstrap_lladao_env.sh"
        exec python -m eval.gui_grounding.summarize_table3_sweep "$@"
    ' bash "${args[@]}"

echo "JSON: ${RESULTS_ROOT}/table3_sweep.json"
echo "CSV:  ${RESULTS_ROOT}/table3_sweep.csv"
