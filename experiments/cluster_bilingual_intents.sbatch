#!/bin/bash
#SBATCH --job-name=intent-retrieval
#SBATCH --partition=gpu
#SBATCH --nodelist=ws5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/media/Ws5_4TB/zdx/expert-find-agent/logs/intent-retrieval-%j.out
#SBATCH --error=/media/Ws5_4TB/zdx/expert-find-agent/logs/intent-retrieval-%j.err
set -euo pipefail
cd /media/Ws5_4TB/zdx/expert-find-agent
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
venv/bin/python -u -m experiments.bilingual.intent_retrieval_diagnostic --output artifacts/bilingual-intent-retrieval-results-v1
