"""Recompute upstream frozen baselines; this does not run our retrieval method."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

COMMIT='8a19258214b2e6973cdc07614908c5caf654fd51'


def run():
    root=Path('artifacts/public/n-clue-upstream')
    if not root.exists():
        subprocess.run(['git','clone','https://github.com/sunggukcha/n_clue.git',str(root)],check=True)
        subprocess.run(['git','-C',str(root),'checkout','--detach',COMMIT],check=True)
    actual=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if actual!=COMMIT:raise ValueError('Expected pinned n-Clue commit; existing checkout left unchanged')
    completed=subprocess.run([sys.executable,'reproduce.py','--verify'],cwd=root,capture_output=True,text=True,check=True)
    summary=json.loads((root/'reproduced_results/summary.json').read_text())
    result=dict(repository='https://github.com/sunggukcha/n_clue',commit=actual,
        script_sha256=hashlib.sha256((root/'reproduce.py').read_bytes()).hexdigest(),
        command='python reproduce.py --verify',stdout=completed.stdout,counts=summary['counts'],
        note='Official frozen-ranking reproduction only. Not newly executed model retrieval, timing, or expert-search quality.',
        code_license='Apache-2.0 upstream',annotation_license='CC-BY-4.0 upstream; source documents not redistributed')
    Path('docs/exploration/nclue-reproduction.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
