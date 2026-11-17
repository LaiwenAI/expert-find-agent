#!/bin/bash
#SBATCH --job-name=fresh-dense-runs
#SBATCH --partition=gpu
#SBATCH --nodelist=ws6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/mnt/sda/dxzhou/zdx/expert-find-agent/logs/fresh-dense-runs-%j.out
#SBATCH --error=/mnt/sda/dxzhou/zdx/expert-find-agent/logs/fresh-dense-runs-%j.err
set -euo pipefail
cd /mnt/sda/dxzhou/zdx/expert-find-agent
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
venv/bin/python -u experiments/cache_fresh_dense_runs.py
