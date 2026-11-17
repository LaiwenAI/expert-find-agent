"""Post-hoc development diagnostic: normalize cached source-only annotations."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from expert_search.evidencebench_data import prepare, aspect_coverage
from expert_search.source_roles import ROLES, normalize_roles, prioritize_roles
from expert_search.statistics import paired_interval


def run():
    parent = Path('artifacts/evidencebench-role-index-v1')
    out = Path('artifacts/evidencebench-role-repair-v1')
    out.mkdir(exist_ok=True)
    data = prepare()
    protocol = dict(scope='Post-hoc development diagnostic, not independent evidence. No new model calls; source-only cached responses. Conflicts and omissions become uncertain, within-role duplicates deduplicate, foreign IDs or invalid shape trigger raw fallback. No oracle repair.',
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    normalization_sha256=hashlib.sha256(Path('src/expert_search/source_roles.py').read_bytes()).hexdigest(),
                    parents={key:hashlib.sha256((parent/(key+'.json')).read_bytes()).hexdigest() for key in data},
                    selection='Full-pool semantic order, stable preference for method/result/interpretation, top10; unknowns retained in original relative order.')
    marker = out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text()) != protocol:
        raise ValueError('Changed repair diagnostic protocol')
    marker.write_text(json.dumps(protocol, indent=2))
    from expert_search.qwen_embedding import QwenEmbedding
    from evidencebench_pilot import TASK
    model = QwenEmbedding(max_length=512, batch_size=8)
    results = []
    for key,row in data.items():
        source = json.loads((parent/(key+'.json')).read_text())
        start = time.perf_counter()
        try:
            payload=json.loads(source['response']['choices'][0]['message']['content'])
            normalized=normalize_roles(payload,len(row['paper_as_candidate_pool']))
            status='normalized'
        except (KeyError, IndexError, TypeError, ValueError):
            normalized=dict(roles={r:[] for r in ROLES}, missing=0, conflicts=0, within_role_duplicates=0)
            status='fallback'
        normalization_seconds=time.perf_counter()-start
        vectors=np.load(Path('artifacts/evidencebench-pilot-v1')/(key+'.npy'),allow_pickle=False)
        start=time.perf_counter()
        q=model.encode(['Instruct: '+TASK+'\nQuery: '+row['hypothesis']])[0]
        scores=vectors@q
        rank=sorted(range(len(scores)),key=lambda i:(-float(scores[i]),i))
        selected=prioritize_roles(rank,normalized['roles'])[:10]
        online_seconds=time.perf_counter()-start
        results.append(dict(id=key,status=status,normalization=normalized,selected=selected,
                            metrics=aspect_coverage(row,selected),baseline=aspect_coverage(row,rank[:10]),
                            normalization_seconds=normalization_seconds,online_seconds=online_seconds))
    report=dict(protocol=protocol,papers=len(results),normalized=sum(r['status']=='normalized' for r in results),
                aspect_recall=float(np.mean([r['metrics']['aspect_recall'] for r in results])),
                difference_vs_qwen=paired_interval([r['metrics']['aspect_recall']-r['baseline']['aspect_recall'] for r in results]),
                missing=sum(r['normalization']['missing'] for r in results),conflicts=sum(r['normalization']['conflicts'] for r in results),
                mean_online_seconds=float(np.mean([r['online_seconds'] for r in results])),
                offline_compilation_seconds=sum(json.loads((parent/(key+'.json')).read_text())['offline_seconds'] for key in data),
                offline_normalization_seconds=sum(r['normalization_seconds'] for r in results),
                warning='New encoding timings include ranking/priority selection but exclude model initialization, source vector loads/encoding. Historical run, not paired interface speed. Offline cost not amortized; labels and role semantics not independently validated; auxiliary evidence coverage only.')
    (out/'evaluation.json').write_text(json.dumps(results,indent=2))
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='protocol'},indent=2))


if __name__=='__main__':run()
