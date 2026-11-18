#!/bin/bash
#SBATCH --job-name=open-decision-pilot
#SBATCH --partition=gpu
#SBATCH --nodelist=ws5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/media/Ws5_4TB/zdx/expert-find-agent/logs/open-decision-pilot-%j.out
#SBATCH --error=/media/Ws5_4TB/zdx/expert-find-agent/logs/open-decision-pilot-%j.err
set -euo pipefail
cd /media/Ws5_4TB/zdx/expert-find-agent
export PYTHONPATH="$PWD/src:$PWD"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
venv/bin/python -u -m experiments.decision.merit_open_decision_pilot
