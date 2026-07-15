#!/usr/bin/env bash
# Copyright 2026 LLaDA-o contributors.
# SPDX-License-Identifier: Apache-2.0

# Clariden GH200 defaults. Command-line sbatch options override these
# directives; adjust the resource requests when using another cluster.
#SBATCH --job-name=lladao-gui-120k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# Slurm executes a spool copy of this script, so BASH_SOURCE points under
# /var/spool instead of the repository. Submit from the repository root or set
# REPO_ROOT explicitly when submitting from another directory.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
ENVIRONMENT_FILE="${ENVIRONMENT_FILE:-${REPO_ROOT}/lladao.toml}"

cd "${REPO_ROOT}"

[[ -f "${ENVIRONMENT_FILE}" ]] || {
  echo "error: EDF file does not exist: ${ENVIRONMENT_FILE}" >&2
  exit 1
}
export NNODES="${SLURM_NNODES}"

if [[ -z "${GPUS_PER_NODE:-}" ]]; then
  gpu_spec="${SLURM_GPUS_ON_NODE:-4}"
  if [[ "${gpu_spec}" =~ ([0-9]+)$ ]]; then
    export GPUS_PER_NODE="${BASH_REMATCH[1]}"
  else
    echo "error: cannot infer GPUS_PER_NODE from SLURM_GPUS_ON_NODE=${gpu_spec}" >&2
    exit 1
  fi
fi
export NPROC_PER_NODE="${NPROC_PER_NODE:-${GPUS_PER_NODE}}"

mapfile -t job_hosts < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
(( ${#job_hosts[@]} > 0 )) || {
  echo "error: Slurm returned no hosts for ${SLURM_JOB_NODELIST}" >&2
  exit 1
}
export MASTER_ADDR="${MASTER_ADDR:-${job_hosts[0]}}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"

threads_per_gpu=$((SLURM_CPUS_PER_TASK / GPUS_PER_NODE))
(( threads_per_gpu > 0 )) || threads_per_gpu=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${threads_per_gpu}}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONFAULTHANDLER=1

echo "Slurm job ${SLURM_JOB_ID}: ${SLURM_NNODES} node(s), ${GPUS_PER_NODE} GPU(s) per node"
echo "Hosts: ${job_hosts[*]}"
echo "Rendezvous: ${MASTER_ADDR}:${MASTER_PORT}"

# Run one launcher task per node. Each launcher creates one training process per
# local GPU and derives its node rank from SLURM_NODEID.
srun \
  --environment="${ENVIRONMENT_FILE}" \
  --network=disable_rdzv_get \
  --nodes="${SLURM_NNODES}" \
  --ntasks="${SLURM_NNODES}" \
  --ntasks-per-node=1 \
  --gpus-per-task="${GPUS_PER_NODE}" \
  --kill-on-bad-exit=1 \
  bash -c 'source "${LLADAO_REPO_ROOT}/scripts/bootstrap_lladao_env.sh" && exec bash "${LLADAO_REPO_ROOT}/scripts/train_gui_grounding_120k.sh"'
