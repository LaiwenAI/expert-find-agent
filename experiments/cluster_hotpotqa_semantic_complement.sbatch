#!/bin/bash
#SBATCH --job-name=hotpotqa-complement
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
set -euo pipefail
workspace=$1
cd "$workspace"
export PYTHONPATH="$PWD/src:$PWD"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
venv/bin/python -u -m experiments.hotpotqa.semantic_complement \
  --dataset artifacts/public/hotpotqa/hotpot_dev_distractor_v1.json \
  --output artifacts/hotpotqa-semantic-complement-v1
