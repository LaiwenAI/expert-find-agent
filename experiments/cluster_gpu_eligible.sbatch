#!/bin/bash
#SBATCH --job-name=gpu-eligible
#SBATCH --partition=gpu
#SBATCH --nodelist=ws5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/media/Ws5_4TB/zdx/expert-find-agent/logs/gpu-eligible-%j.out
#SBATCH --error=/media/Ws5_4TB/zdx/expert-find-agent/logs/gpu-eligible-%j.err
set -euo pipefail
cd /media/Ws5_4TB/zdx/expert-find-agent
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
venv/bin/python -u -m experiments.bilingual.gpu_eligible_diagnostic --output artifacts/bilingual-gpu-eligible-v1
