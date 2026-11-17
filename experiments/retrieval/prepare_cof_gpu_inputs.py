"""Verify official CoF source members and freeze cached GPU benchmark inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from expert_search.cof_data import SHA256
from expert_search.cof_benchmark import load


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(output):
    root,papers,reviewers,labels=load('SIGIR');archive=Path('artifacts/public/cof/data.zip')
    if sha(archive)!=SHA256:raise ValueError('Changed official archive')
    names=('SIGIR_papers_test.json','SIGIR_reviewers_test.json','SIGIR_queries_test_soft.json','SIGIR_queries_test_hard.json')
    with zipfile.ZipFile(archive) as source:
        for name in names:
            if hashlib.sha256(source.read('data/'+name)).hexdigest()!=sha(root/name):raise ValueError('Extracted data differs from official archive')
    vector=root/'qwen-embedding-torch-v1';summary=json.loads((vector/'summary.json').read_text())
    if sha(vector/'vectors.npy')!=summary['vectors_sha256'] or set(labels['soft'])!=set(labels['hard']) or len(labels['soft'])!=73:raise ValueError('Changed vectors or labels')
    lexical=root/'unique-profile-corpus';qids=sorted(labels['soft'])
    paths=[root/name for name in names]+[vector/'protocol.json',vector/'summary.json',vector/'vectors.npy',lexical/'evidence.jsonl',lexical/'experts.jsonl',lexical/'fast.sqlite']+[vector/(qid+'.json') for qid in qids]
    output.mkdir(parents=True,exist_ok=False)
    report={'dataset':'SIGIR','archive_sha256':sha(archive),'query_ids':qids,'input_sha256':{str(p):sha(p) for p in paths},
            'code_sha256':sha(Path(__file__)),'scope':'Official archive members checked byte-for-byte; cached model vectors and lexical corpus frozen by hash. Public reviewer matching is not industrial expert-fit relevance. Upstream data and vectors remain outside the source release.'}
    (output/'input-provenance.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'queries':len(qids),'archive_sha256':report['archive_sha256'],'files':len(paths)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
