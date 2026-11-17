#!/bin/bash
#SBATCH --job-name=bilingual-retrieval
#SBATCH --partition=gpu
#SBATCH --nodelist=ws6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/sda/dxzhou/zdx/expert-find-agent/logs/bilingual-%j.out
#SBATCH --error=/mnt/sda/dxzhou/zdx/expert-find-agent/logs/bilingual-%j.err
set -euo pipefail
cd /mnt/sda/dxzhou/zdx/expert-find-agent
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
# Requires the frozen query/selection files and code on this compute disk.
# The script verifies all source/vector/checkpoint hashes before model work.
venv/bin/python -u -m experiments.bilingual.retrieval_diagnostic
