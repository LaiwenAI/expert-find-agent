"""Freeze remaining translated public queries for compute-node encoding."""
import hashlib
import json
from pathlib import Path
from expert_search.cof_benchmark import load


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root,papers,_,_=load('SIGIR');folder=Path('artifacts/cof-sigir-bilingual-remaining-v1')
    protocol=json.loads((folder/'protocol.json').read_text());summary=json.loads((folder/'summary.json').read_text())
    if sha(folder/'protocol.json')!=summary['protocol_sha256'] or any(sha(root/n)!=h for n,h in protocol['source_sha256'].items()):raise ValueError('Changed source or translation protocol')
    rows=[];missing=[]
    for qid in protocol['query_ids']:
        name=hashlib.sha256(qid.encode()).hexdigest()+'.json';path=folder/'queries'/name
        if sha(path)!=summary['record_sha256'][name]:raise ValueError('Changed translation record')
        record=json.loads(path.read_text());paper=papers[qid];english=(paper.get('title','')+'\n'+paper.get('abstract','')).strip()
        if record['query_id']!=qid or hashlib.sha256(english.encode()).hexdigest()!=record['english_sha256']:raise ValueError('Changed query identity')
        if record['chinese'] is None:missing.append(qid);continue
        if hashlib.sha256(record['chinese'].encode()).hexdigest()!=record['chinese_sha256']:raise ValueError('Changed translation')
        has_han=any('\u4e00'<=c<='\u9fff' for c in record['chinese'])
        rows.append({'id':qid,'english':english,'chinese':record['chinese'],'status':record['status'],'contains_han':has_han})
    if len(rows)+len(missing)!=len(protocol['query_ids']) or len({r['id'] for r in rows})!=len(rows):raise ValueError('Incomplete selection')
    target=folder/'encode-input.jsonl';marker=folder/'encode-manifest.json'
    if target.exists() or marker.exists():raise FileExistsError('Encoding selection already exists')
    target.write_text(''.join(json.dumps(r,ensure_ascii=True)+'\n' for r in rows))
    report={'all_query_ids':protocol['query_ids'],'available_query_ids':[r['id'] for r in rows],'missing_query_ids':missing,'available':len(rows),
            'han_present':sum(r['contains_han'] for r in rows),'model_screened':sum(r['status']=='model_screened' and r['contains_han'] for r in rows),
            'queries_sha256':sha(target),'translation_protocol_sha256':sha(folder/'protocol.json'),'translation_summary_sha256':sha(folder/'summary.json'),
            'checkpoint_provenance_sha256':sha('artifacts/models/qwen3-embedding-0.6b/checkpoint-provenance.json'),
            'encoder_sha256':sha('src/expert_search/qwen_embedding_torch.py'),'task_module_sha256':sha('src/expert_search/qwen_embedding.py'),
            'encoder_script_sha256':sha('experiments/bilingual/cof_query_encode.py'),
            'scope':'Public CoF translated query views; complete-English translations and same-model screens are not human gold. Missing translations remain missing; Han presence is not a language-quality certificate.'}
    marker.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
