"""Check transferred vectors and post-hoc Slurm/checkpoint provenance."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(folder):
    summary=json.loads((folder/'encoding-summary.json').read_text());selection=json.loads((folder/'encode-manifest.json').read_text())
    if sha(folder/'query-vectors.npz')!=summary['vectors_sha256'] or sha(folder/'encode-input.jsonl')!=selection['queries_sha256'] or sha(folder/'encode-manifest.json')!=summary['selection_sha256']:
        raise ValueError('Changed transferred encoding artifacts')
    job=summary['slurm_job_id']
    if not isinstance(job,str) or not re.fullmatch(r'\d+',job) or summary['node']!='ws5':raise ValueError('Unexpected compute allocation')
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=6','dxzhou@10.21.50.60']
    account=subprocess.check_output(ssh+[f'sacct -j {job} --format=JobIDRaw,State,ExitCode --parsable2 --noheader'],text=True,timeout=20)
    lines=[line.strip().split('|') for line in account.splitlines() if line.strip()]
    if not any(len(line)>=3 and line[0]==job and line[1]=='COMPLETED' and line[2]=='0:0' for line in lines):raise ValueError('Job not confirmed successful')
    command="srun --partition=gpu --nodelist=ws5 --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=1G --time=00:02:00 sha256sum /media/Ws5_4TB/zdx/expert-find-agent/artifacts/models/qwen3-embedding-0.6b/checkpoint-provenance.json"
    remote=subprocess.check_output(ssh+[command],text=True,timeout=60).split()[0];local=sha('artifacts/models/qwen3-embedding-0.6b/checkpoint-provenance.json')
    if remote!=local:raise ValueError('Post-hoc checkpoint provenance mismatch')
    out=folder/'transfer-verification.json'
    if out.exists():raise FileExistsError('Do not overwrite verification')
    report={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'slurm_job_id':job,'slurm_state':'COMPLETED','slurm_exit_code':'0:0',
            'vectors_sha256':sha(folder/'query-vectors.npz'),'encoding_summary_sha256':sha(folder/'encoding-summary.json'),
            'selection_sha256':sha(folder/'encode-manifest.json'),'local_checkpoint_provenance_sha256':local,'remote_checkpoint_provenance_posthoc_sha256':remote,
            'pinned_provenance_in_job':'checkpoint_provenance_sha256' in summary,'code_sha256':sha(Path(__file__)),
            'scope':'Post-hoc Slurm and node-side provenance-file check; the initial encoding job verified weights against that node-side file but did not itself pin the provenance file hash. Later file checks cannot prove absence of intervening changes.'}
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',type=Path,required=True);a=p.parse_args();main(a.folder)
