#!/bin/bash
#SBATCH --job-name=fresh-demands
#SBATCH --partition=gpu
#SBATCH --nodelist=ws5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/media/Ws5_4TB/zdx/expert-find-agent/logs/fresh-demands-%j.out
#SBATCH --error=/media/Ws5_4TB/zdx/expert-find-agent/logs/fresh-demands-%j.err
set -euo pipefail
cd /media/Ws5_4TB/zdx/expert-find-agent
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
venv/bin/python -u experiments/fresh_method_retrieval.py --stage encode
