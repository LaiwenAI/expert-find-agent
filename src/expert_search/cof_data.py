"""Pinned CoF public paper-to-reviewer data; original soft/hard labels retained."""
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

URL='https://drive.usercontent.google.com/download?id=1IbQBVrpu3wMXahkqKVFDDYlB5_nS5cTx&export=download&confirm=t'
SHA256='a41e8a5aef6d1c38db5b159e342416a789e9799deb10b6d51a398e7479d360d9'
DATASETS=('SIGIR','NIPS','SciRepEval','KDD')


def unique_records(rows,key):
    output={}
    for row in rows:
        identity=row[key]
        if identity in output and output[identity]!=row:raise ValueError('Conflicting duplicate source ID')
        output[identity]=row
    return output


def prepare(dataset='SIGIR',root='artifacts/public/cof'):
    if dataset not in DATASETS:raise ValueError('Unknown CoF dataset')
    root=Path(root);root.mkdir(parents=True,exist_ok=True);archive=root/'data.zip'
    if not archive.exists():
        partial=root/'data.zip.partial'
        if partial.exists():raise ValueError('Partial download exists; inspect its live process before restarting')
        with urllib.request.urlopen(URL,timeout=30) as response,partial.open('wb') as target:
            while True:
                block=response.read(1024*1024)
                if not block:break
                target.write(block)
        if hashlib.sha256(partial.read_bytes()).hexdigest()!=SHA256:raise ValueError('Unexpected CoF data archive')
        partial.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=SHA256:raise ValueError('Changed CoF archive')
    out=root/dataset;out.mkdir(exist_ok=True);raw={}
    with zipfile.ZipFile(archive) as source:
        for kind in ('papers','reviewers','queries_soft','queries_hard'):
            filename=(f'{dataset}_queries_test_{kind[8:]}.json' if kind.startswith('queries_') else f'{dataset}_{kind}_test.json')
            body=source.read('data/'+filename);raw[kind]=[json.loads(line) for line in body.splitlines() if line.strip()]
            (out/filename).write_bytes(body)
    papers=unique_records(raw['papers'],'paper');reviewers=unique_records(raw['reviewers'],'reviewer')
    if any(any(p not in papers for p in r['papers']) for r in reviewers.values()):
        raise ValueError('Missing publication profile references')
    query_ids={r['query_id'] for r in raw['queries_soft']}
    if query_ids!={r['query_id'] for r in raw['queries_hard']} or not query_ids.issubset(papers):raise ValueError('Query mismatch')
    for task in ('soft','hard'):
        for q in raw['queries_'+task]:
            if set(q['score'])-set(reviewers) or any(type(v)!=int or v not in (0,1) for v in q['score'].values()):
                raise ValueError('Invalid binary candidate labels')
    empty_profiles={eid for eid,r in reviewers.items() if not r['papers']}
    judged_reviewers={eid for task in ('soft','hard') for q in raw['queries_'+task] for eid in q['score']}
    if empty_profiles&judged_reviewers:raise ValueError('Judged reviewer has an empty profile')
    query_texts={qid:(papers[qid].get('title','')+'\n'+papers[qid].get('abstract','')).strip().casefold() for qid in query_ids}
    profile_ids={p for r in reviewers.values() for p in r['papers']}
    profile_texts={(papers[pid].get('title','')+'\n'+papers[pid].get('abstract','')).strip().casefold() for pid in profile_ids}
    manifest=dict(dataset=dataset,archive_url=URL,archive_sha256=SHA256,source='https://github.com/yuz9yuz/CoF',
        papers=len(papers),reviewers=len(reviewers),queries=len(query_ids),profile_papers=len(profile_ids),
        identical_duplicate_paper_rows=len(raw['papers'])-len(papers),
        empty_unjudged_profiles=sorted(empty_profiles),
        judged_reviewers=len({eid for q in raw['queries_soft'] for eid in q['score']}),
        labeled_pairs={task:sum(len(q['score']) for q in raw['queries_'+task]) for task in ('soft','hard')},
        all_reviewers_judged_per_query={task:all(set(q['score'])==set(reviewers) for q in raw['queries_'+task]) for task in ('soft','hard')},
        query_ids_in_profiles=sorted(query_ids&profile_ids),exact_query_texts_in_profiles=sorted(qid for qid,t in query_texts.items() if t in profile_texts),
        label_note='Original released binary soft/hard labels, not newly human-judged pair labels. SIGIR labels derive from expertise aspects; inspect original source protocol. Unlisted candidates remain unjudged.',
        publication_note='Paper-to-reviewer task, not industrial demand relevance. Code Apache-2.0 upstream; dataset/source-text rights are separate and no third-party data enters our source bundle.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest


if __name__=='__main__':
    for dataset in DATASETS:print(json.dumps(prepare(dataset),indent=2))
