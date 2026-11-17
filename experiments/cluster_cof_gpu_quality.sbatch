#!/bin/bash
#SBATCH --job-name=cof-gpu-quality
#SBATCH --partition=gpu
#SBATCH --nodelist=ws5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/media/Ws5_4TB/zdx/expert-find-agent/logs/cof-gpu-quality-%j.out
#SBATCH --error=/media/Ws5_4TB/zdx/expert-find-agent/logs/cof-gpu-quality-%j.err
set -euo pipefail
cd /media/Ws5_4TB/zdx/expert-find-agent
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
venv/bin/python -u -m experiments.retrieval.cof_gpu_quality_diagnostic --output artifacts/cof-gpu-quality-v1
