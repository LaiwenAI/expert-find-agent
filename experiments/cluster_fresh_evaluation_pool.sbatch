#!/bin/bash
#SBATCH --job-name=fresh-eval-pool
#SBATCH --partition=gpu
#SBATCH --nodelist=ws6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/sda/dxzhou/zdx/expert-find-agent/logs/fresh-eval-pool-%j.out
#SBATCH --error=/mnt/sda/dxzhou/zdx/expert-find-agent/logs/fresh-eval-pool-%j.err
set -euo pipefail
cd /mnt/sda/dxzhou/zdx/expert-find-agent
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
venv/bin/python -u experiments/fresh_evaluation_pool.py
