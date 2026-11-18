#!/bin/bash
#SBATCH --job-name=open-decision-model
#SBATCH --partition=gpu
#SBATCH --nodelist=ws5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/media/Ws5_4TB/zdx/expert-find-agent/logs/open-decision-model-%j.out
#SBATCH --error=/media/Ws5_4TB/zdx/expert-find-agent/logs/open-decision-model-%j.err
set -euo pipefail
cd /media/Ws5_4TB/zdx/expert-find-agent
export HF_HOME="$PWD/artifacts/huggingface"
venv/bin/python -u - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
import hashlib, json
revision='cdbee75f17c01a7cc42f958dc650907174af0554'
root=Path('artifacts/models/qwen3-4b-instruct-2507')
snapshot_download(repo_id='Qwen/Qwen3-4B-Instruct-2507',revision=revision,local_dir=root,
                  allow_patterns=['*.json','*.safetensors','*.txt','*.model','LICENSE'])
files={}
for p in sorted(root.rglob('*')):
    if p.is_file() and '.cache' not in p.parts:
        h=hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        files[str(p.relative_to(root))]=h.hexdigest()
if not any(x.endswith('.safetensors') for x in files):raise RuntimeError('Model weights missing')
(root/'local-provenance.json').write_text(json.dumps({'repo':'Qwen/Qwen3-4B-Instruct-2507','revision':revision,'files':files},indent=2)+'\n')
print('Model snapshot files:',len(files),'revision:',revision,flush=True)
PY
