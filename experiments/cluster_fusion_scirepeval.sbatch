#!/bin/bash
#SBATCH --job-name=fusion-SciRepEval
#SBATCH --partition=gpu
#SBATCH --nodelist=ws6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/sda/dxzhou/zdx/expert-find-agent/logs/fusion-SciRepEval-%j.out
#SBATCH --error=/mnt/sda/dxzhou/zdx/expert-find-agent/logs/fusion-SciRepEval-%j.err
set -euo pipefail
cd /mnt/sda/dxzhou/zdx/expert-find-agent
export HF_HOME="$PWD/cache/huggingface"
export PIP_CACHE_DIR="$PWD/cache/pip"
export TMPDIR="$PWD/tmp"
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
venv/bin/python - <<'PY'
import hashlib,json,os,platform
from pathlib import Path
import torch,transformers
root=Path('artifacts/models/qwen3-embedding-0.6b')
manifest=json.loads((root/'checkpoint-provenance.json').read_text())
for name,expected in manifest['files'].items():
    with (root/name).open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest!=expected:raise ValueError('Changed checkpoint file: '+name)
if not torch.cuda.is_available():raise RuntimeError('No allocated CUDA GPU')
x=torch.ones((32,32),device='cuda');torch.cuda.synchronize()
report=dict(job_id=os.environ['SLURM_JOB_ID'],node=platform.node(),torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),cuda_smoke=float((x@x).sum()))
Path('artifacts/cluster-runtime-'+os.environ['SLURM_JOB_ID']+'.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
PY
venv/bin/python -u experiments/cof_qwen_embedding.py --dataset SciRepEval --backend torch
venv/bin/python -u experiments/cof_source_fusion.py --dataset SciRepEval --lexical-snapshot artifacts/lexical-snapshots-v1/SciRepEval.npz
