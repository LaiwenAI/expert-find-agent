#!/bin/bash
#SBATCH --job-name=owned-effort
#SBATCH --partition=gpu
#SBATCH --nodelist=ws6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/sda/dxzhou/zdx/expert-find-agent/logs/owned-effort-%j.out
#SBATCH --error=/mnt/sda/dxzhou/zdx/expert-find-agent/logs/owned-effort-%j.err
set -euo pipefail
cd /mnt/sda/dxzhou/zdx/expert-find-agent
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
venv/bin/python -u experiments/global_owned_effort.py
