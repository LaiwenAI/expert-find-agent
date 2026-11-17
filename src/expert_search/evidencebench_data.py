"""Pinned EvidenceBench adapter; evidence extraction, not expert-fit gold."""
import hashlib
import json
import urllib.request
from pathlib import Path

REVISION='bf1d9633c694381c7b016fd56ee9f95f48593cc3'
FILES={'dev':('datasets/evidencebench_dev_set.json','189700f6f467e280ff233c82f1697fb7ceb11a79'),
       'test':('datasets/evidencebench_test_set.json','df380a1ba1359f9cea8bca2f2298dc9fd99e6513')}


def prepare(split='dev',directory='artifacts/public/evidencebench'):
    if split not in FILES:raise ValueError('Unknown EvidenceBench split')
    path,blob=FILES[split];root=Path(directory);root.mkdir(parents=True,exist_ok=True);target=root/(split+'.json')
    url='https://raw.githubusercontent.com/EvidenceBench/EvidenceBench/'+REVISION+'/'+path
    if not target.exists():
        partial=target.with_suffix('.partial')
        if partial.exists():raise ValueError('Partial EvidenceBench download; inspect before replacing')
        with urllib.request.urlopen(url,timeout=60) as response,partial.open('wb') as stream:
            while True:
                block=response.read(1024*1024)
                if not block:break
                stream.write(block)
        body=partial.read_bytes()
        if hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()!=blob:raise ValueError('Unexpected EvidenceBench download')
        partial.replace(target)
    body=target.read_bytes()
    if hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()!=blob:raise ValueError('Changed EvidenceBench snapshot')
    rows=json.loads(body)
    for row in rows.values():
        n=len(row['paper_as_candidate_pool'])
        if not n or any(not isinstance(s,str) for s in row['paper_as_candidate_pool']):raise ValueError('Invalid sentence inventory')
        if any(not 0<=int(i)<n for i in row['sentence_index2aspects']):raise ValueError('Invalid annotation sentence index')
        if any(not set(v)<=set(row['aspect_list_ids']) for v in row['sentence_index2aspects'].values()):raise ValueError('Unknown annotated aspect')
    spec=dict(revision=REVISION,url=url,git_blob_sha1=blob,sha256=hashlib.sha256(body).hexdigest(),split=split,records=len(rows),
              scope='Original upstream hypothesis-to-paper-sentence evidence task. Annotations generated with expert-guided validation; not all sentences newly human labeled, and not enterprise expert-fit gold.',
              license='Original dev: CC-BY-NC-SA; original test: CC-BY. Upstream license retained separately; adapter does not relicense source text.')
    (root/(split+'-manifest.json')).write_text(json.dumps(spec,indent=2));return rows


def aspect_coverage(row,selected):
    if not isinstance(selected,(list,tuple)) or len(set(selected))!=len(selected) or any(type(i) is not int or not 0<=i<len(row['paper_as_candidate_pool']) for i in selected):raise ValueError('Invalid evidence sentence selection')
    gold=set().union(*(set(a) for a in row['sentence_index2aspects'].values()))
    covered=set().union(*(set(row['sentence_index2aspects'].get(str(i),[])) for i in selected))
    if not gold:raise ValueError('No annotated aspects')
    return dict(aspect_recall=len(covered)/len(gold),covered_aspects=len(covered),total_aspects=len(gold),selected_sentences=len(selected))
