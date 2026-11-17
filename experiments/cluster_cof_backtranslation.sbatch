#!/bin/bash
#SBATCH --job-name=cof-backtranslation
#SBATCH --partition=gpu
#SBATCH --nodelist=ws5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/media/Ws5_4TB/zdx/expert-find-agent/logs/cof-backtranslation-%j.out
#SBATCH --error=/media/Ws5_4TB/zdx/expert-find-agent/logs/cof-backtranslation-%j.err
set -euo pipefail
cd /media/Ws5_4TB/zdx/expert-find-agent
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
venv/bin/python -u -m experiments.bilingual.cof_query_encode --folder artifacts/cof-sigir-backtranslation-v1
