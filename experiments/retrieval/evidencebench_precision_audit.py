"""Post-hoc sentence-precision and results-aspect audit of frozen selections."""
import hashlib
import json
import statistics
from pathlib import Path

from expert_search.evidencebench_data import aspect_coverage
from expert_search.statistics import clustered_paired_interval


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(row,selected):
    aspect_coverage(row,selected)
    if len(selected)!=10:raise ValueError('Expected exactly ten sentences')
    mapping=row['sentence_index2aspects'];result_ids=set(row['results_aspect_list_ids'] or [])
    positive=sum(bool(mapping.get(str(i),[])) for i in selected)
    covered=set().union(*(set(mapping.get(str(i),[])) for i in selected))
    return {'sentence_precision_at_10':positive/10,
            'results_aspect_recall':len(covered&result_ids)/len(result_ids) if result_ids else None}


def main():
    root=Path('artifacts/evidencebench-role-test-v1');data_path=Path('artifacts/public/evidencebench/test.json')
    p=json.loads((root/'protocol.json').read_text())
    if sha(data_path)!=p['dataset_sha256']:raise ValueError('Changed dataset')
    data=json.loads(data_path.read_text());rows=json.loads((root/'evaluation.json').read_text())
    if len(rows)!=len(data) or {r['id'] for r in rows}!=set(data):raise ValueError('Incomplete or duplicate results')
    results=[]
    for row in rows:
        source=data[row['id']]
        if aspect_coverage(source,row['selected'])!=row['metrics'] or aspect_coverage(source,row['baseline_selected'])!=row['baseline']:raise ValueError('Prior metrics mismatch')
        results.append({'paper_id':source['paper_id'],'role':metrics(source,row['selected']),'baseline':metrics(source,row['baseline_selected'])})
    report={}
    for metric in ('sentence_precision_at_10','results_aspect_recall'):
        usable=[r for r in results if r['role'][metric] is not None and r['baseline'][metric] is not None]
        differences=[r['role'][metric]-r['baseline'][metric] for r in usable]
        report[metric]={'queries':len(usable),'baseline_mean':statistics.mean(r['baseline'][metric] for r in usable),
                        'role_mean':statistics.mean(r['role'][metric] for r in usable),
                        'paired_difference':clustered_paired_interval(differences,[r['paper_id'] for r in usable])}
    result={'queries':len(rows),'metrics':report,'dataset_sha256':sha(data_path),'protocol_sha256':sha(root/'protocol.json'),
            'evaluation_sha256':sha(root/'evaluation.json'),'code_sha256':sha(Path(__file__)),
            'scope':'Post-hoc analysis of original model-assisted EvidenceBench aspect labels, not human gold or enterprise expert fit. Sentence precision means a selected sentence covers at least one annotated aspect; it does not measure truth, attribution or independence. Results-aspect recall excludes queries without results aspects. Paper-cluster bootstrap is unadjusted for multiple analyses and does not include model or annotation uncertainty.'}
    out=Path('artifacts/evidencebench-precision-audit-v1');out.mkdir(exist_ok=False);(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
