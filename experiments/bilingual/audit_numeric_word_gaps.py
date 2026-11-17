"""Aggregate unparsed word-quantity flags on the frozen bilingual construction set."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

from expert_search.bilingual.numeric_word_audit import audit


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path('artifacts/dataset-release/bilingual-query-construction-v1');manifest=json.loads((root/'manifest.json').read_text())
    source=root/'synthetic-queries.jsonl'
    if sha(source)!=manifest['outputs_sha256'][source.name]:raise ValueError('Changed synthetic query source')
    rows=[json.loads(line) for line in source.read_text().splitlines()]
    if len(rows)!=5000 or len({r['id'] for r in rows})!=5000 or any(r['split']!='construction' for r in rows):raise ValueError('Unexpected cohort')
    counts=Counter();by_type=defaultdict(Counter)
    for row in rows:
        result=audit(row['zh'],row['en']);flag=result['source_word_quantity_flag'] or result['target_word_quantity_flag']
        counts['pairs']+=1;counts['flagged_either']+=flag;counts['flagged_source']+=result['source_word_quantity_flag'];counts['flagged_target']+=result['target_word_quantity_flag']
        counts['v2_match_downgraded']+=flag and result['v2_status']=='match'
        counts['current_screened_flagged']+=flag and row['status']=='retained_model_screened_pair'
        by_type[row['intent_type']]['pairs']+=1;by_type[row['intent_type']]['flagged']+=flag
    output=Path('artifacts/bilingual-numeric-word-audit-v2');output.mkdir(parents=True,exist_ok=False)
    report={'counts':dict(counts),'by_intent':{k:dict(v) for k,v in sorted(by_type.items())},'source_sha256':sha(source),'manifest_sha256':sha(root/'manifest.json'),
            'code_sha256':{str(p):sha(p) for p in (Path(__file__),Path('src/expert_search/bilingual/numeric_word_audit.py'),Path('src/expert_search/bilingual/postvalidation.py'))},
            'scope':'Conservative literal flags for selected unparsed word quantities. Flagged cases require human inspection, not automatic error labels; many unparsed forms and semantic errors remain undetected. Original status and data remain unchanged. Counts include multiple synthetic views per parent and are not independent quality estimates.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
