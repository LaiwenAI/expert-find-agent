#!/bin/bash
#SBATCH --job-name=evidence-semantic-gaps
#SBATCH --partition=gpu
#SBATCH --nodelist=ws5
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/media/Ws5_4TB/zdx/expert-find-agent/logs/evidence-semantic-gaps-%j.out
#SBATCH --error=/media/Ws5_4TB/zdx/expert-find-agent/logs/evidence-semantic-gaps-%j.err
set -euo pipefail
cd /media/Ws5_4TB/zdx/expert-find-agent
export PYTHONPATH="$PWD/src:$PWD"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
venv/bin/python -u -m experiments.retrieval.evidencebench_semantic_gaps --output artifacts/evidencebench-semantic-gaps-v1
