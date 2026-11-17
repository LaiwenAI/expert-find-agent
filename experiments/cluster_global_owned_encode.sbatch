#!/bin/bash
#SBATCH --job-name=owned-queries
#SBATCH --partition=gpu
#SBATCH --nodelist=ws5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/media/Ws5_4TB/zdx/expert-find-agent/logs/owned-queries-%j.out
#SBATCH --error=/media/Ws5_4TB/zdx/expert-find-agent/logs/owned-queries-%j.err
set -euo pipefail
cd /media/Ws5_4TB/zdx/expert-find-agent
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
venv/bin/python -u experiments/global_owned_routing.py --stage encode
