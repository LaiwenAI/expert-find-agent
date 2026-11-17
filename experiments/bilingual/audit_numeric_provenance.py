"""Audit derived numeric values against cited and complete enterprise sources."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from expert_search.bilingual.numeric_provenance import audit_numeric_provenance


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(root,output):
    manifest=json.loads((root/'manifest.json').read_text());real_path=root/'real-queries.jsonl';derived_path=root/'synthetic-queries.jsonl'
    if any(sha(p)!=manifest['outputs_sha256'][p.name] for p in (real_path,derived_path)):raise ValueError('Changed source export')
    real=list(map(json.loads,real_path.read_text().splitlines()));parents={r['id']:r for r in real};rows=list(map(json.loads,derived_path.read_text().splitlines()))
    if len(parents)!=manifest['real_original_queries'] or len(rows)!=manifest['synthetic_slots'] or len({r['id'] for r in rows})!=len(rows):raise ValueError('Changed export inventory')
    output.mkdir(parents=True,exist_ok=False);cases=[]
    for row in rows:
        parent=parents[row['parent_id']]
        if parent['split']!='construction' or row['split']!='construction' or parent['group_id']!=row['group_id'] or row['id']!=row['parent_id']+':'+row['intent_type']:
            raise ValueError('Changed parent binding')
        audits={lang:audit_numeric_provenance(parent['zh'],row['source_ids'],row[lang]) for lang in ('zh','en')}
        cases.append({'id':row['id'],'intent_type':row['intent_type'],'status':row['status'],
                      'checks':{lang:{k:v for k,v in value.items() if k!='scope'} for lang,value in audits.items()}})
    (output/'private-cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    def aggregate(subset):
        result=Counter(pairs=len(subset))
        for case in subset:
            for lang,value in case['checks'].items():
                result[lang+'_has_numeric']+=value['detected_query_values']>0
                result[lang+'_absent_from_cited']+=value['elsewhere_in_parent_values']+value['absent_from_parent_values']>0
                result[lang+'_absent_from_parent']+=value['absent_from_parent_values']>0
            result['either_absent_from_parent']+=any(x['absent_from_parent_values']>0 for x in case['checks'].values())
        return dict(result)
    report={'overall':aggregate(cases),'by_intent':{t:aggregate([r for r in cases if r['intent_type']==t]) for t in manifest['taxonomy']},
            'by_status':{t:aggregate([r for r in cases if r['status']==t]) for t in sorted({r['status'] for r in cases})},
            'dataset_manifest_sha256':sha(root/'manifest.json'),'real_queries_sha256':sha(real_path),'synthetic_queries_sha256':sha(derived_path),
            'code_sha256':{str(p):sha(p) for p in (Path(__file__),Path('src/expert_search/bilingual/numeric_provenance.py'),Path('src/expert_search/bilingual/constraints.py'))},
            'cases_sha256':sha(output/'private-cases.json'),
            'scope':'Flag normalized numeric values absent from cited segments or full Chinese parent. Values and IDs remain private. Numeric extraction is incomplete and cannot validate units, object, translation, arithmetic, source scope or expert relevance; flags are review candidates, not hallucination labels. No data or screening status is changed.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('artifacts/dataset-release/bilingual-query-construction-v1'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.root,a.output)
