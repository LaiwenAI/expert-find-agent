#!/bin/bash
#SBATCH --job-name=evidence-test
#SBATCH --partition=gpu
#SBATCH --nodelist=ws5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/media/Ws5_4TB/zdx/expert-find-agent/logs/evidence-test-%j.out
#SBATCH --error=/media/Ws5_4TB/zdx/expert-find-agent/logs/evidence-test-%j.err
set -euo pipefail
cd /media/Ws5_4TB/zdx/expert-find-agent
export PYTHONPATH="$PWD/src"
export HF_HOME="$PWD/cache/huggingface"
export TMPDIR="$PWD/tmp"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
mkdir -p "$HF_HOME" "$TMPDIR"
venv/bin/python - <<'PY'
import hashlib,json,os,platform
from pathlib import Path
import torch,transformers
root=Path('artifacts/models/qwen3-embedding-0.6b')
for name,expected in json.loads((root/'checkpoint-provenance.json').read_text())['files'].items():
    with (root/name).open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=expected:raise ValueError('Changed checkpoint: '+name)
if not torch.cuda.is_available():raise RuntimeError('No allocated GPU')
report=dict(job_id=os.environ['SLURM_JOB_ID'],node=platform.node(),torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())
Path('artifacts/cluster-runtime-'+os.environ['SLURM_JOB_ID']+'.json').write_text(json.dumps(report,indent=2))
print(report,flush=True)
PY
venv/bin/python -u experiments/evidencebench_role_test.py --stage vectors
