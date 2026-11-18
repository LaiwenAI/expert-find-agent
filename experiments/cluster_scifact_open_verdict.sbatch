#!/bin/bash
#SBATCH --job-name=scifact-open-verdict
#SBATCH --partition=gpu
#SBATCH --nodelist=ws5
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/media/Ws5_4TB/zdx/expert-find-agent/logs/scifact-open-verdict-%j.out
#SBATCH --error=/media/Ws5_4TB/zdx/expert-find-agent/logs/scifact-open-verdict-%j.err
set -euo pipefail
cd /media/Ws5_4TB/zdx/expert-find-agent
export PYTHONPATH="$PWD/src:$PWD"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
venv/bin/python -u -m experiments.scifact.open_verdict_evidence_ablation --cross-order --output artifacts/scifact-open-verdict-factorial-v1
