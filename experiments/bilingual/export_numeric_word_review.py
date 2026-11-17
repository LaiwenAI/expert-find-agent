"""Export a local blind supplement for unrepresented word-quantity flags."""
import hashlib
import json
import random
from pathlib import Path

from expert_search.bilingual.numeric_word_audit import audit
from .audit_review_numeric_coverage import run, verified
from .check_grounding_review import check


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    run()  # Reconstruct and verify the original stratified sample before exclusion.
    data=Path('artifacts/dataset-release/bilingual-query-construction-v1')
    review=Path('artifacts/stratified-grounding-human-review-v1')
    earlier=Path('artifacts/bilingual-numeric-word-audit-v2/summary.json')
    manifest=json.loads((data/'manifest.json').read_text());previous=json.loads(earlier.read_text())
    if sha(data/'manifest.json')!=previous['manifest_sha256'] or sha(data/'synthetic-queries.jsonl')!=previous['source_sha256']:
        raise ValueError('Changed audit inputs')
    for path,digest in previous['code_sha256'].items():
        if sha(path)!=digest:raise ValueError('Changed audit code')
    rows=[json.loads(x) for x in verified(data/'synthetic-queries.jsonl',manifest['outputs_sha256']['synthetic-queries.jsonl']).splitlines()]
    parents={r['id']:r for r in map(json.loads,verified(data/'real-queries.jsonl',manifest['outputs_sha256']['real-queries.jsonl']).splitlines())}
    original={r['query_id'] for r in json.loads((review/'private-mapping.json').read_text())}
    flagged=[]
    for row in rows:
        result=audit(row['zh'],row['en'])
        if result['source_word_quantity_flag'] or result['target_word_quantity_flag']:flagged.append(row)
    if len(flagged)!=previous['counts']['flagged_either'] or len({r['id'] for r in flagged})!=len(flagged):raise ValueError('Changed flag cohort')
    eligible=sorted((r for r in flagged if r['id'] not in original),key=lambda r:r['id'])
    if any(r['split']!='construction' or r['parent_id'] not in parents for r in eligible):raise ValueError('Invalid parent or split')
    rng=random.Random(20261122);rng.shuffle(eligible);forms=[];mapping=[]
    for index,row in enumerate(eligible,1):
        case=f'word-{index:03d}'
        forms.append({'case_id':case,'intent_type':row['intent_type'],'original_need':parents[row['parent_id']]['zh'],
                      'chinese_request':row['zh'],'english_request':row['en'],'chinese_grounding':None,
                      'english_grounding':None,'bilingual_equivalence':None,'rationale':None,'reviewer_id':None})
        mapping.append({'case_id':case,'query_id':row['id'],'parent_id':row['parent_id'],'status':row['status']})
    output=Path('artifacts/numeric-word-grounding-human-review-v1');output.mkdir(parents=True,exist_ok=False)
    payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in forms).encode();private=(json.dumps(mapping,indent=2)+'\n').encode()
    (output/'blind-review.jsonl').write_bytes(payload);(output/'private-mapping.json').write_bytes(private)
    report={'cases':len(forms),'flagged_population':len(flagged),'already_in_original_sample':len(flagged)-len(eligible),'seed':20261122,
            'dataset_manifest_sha256':sha(data/'manifest.json'),'original_sample_manifest_sha256':sha(review/'manifest.json'),
            'audit_sha256':sha(earlier),'output_sha256':hashlib.sha256(payload).hexdigest(),'private_mapping_sha256':hashlib.sha256(private).hexdigest(),
            'code_sha256':sha(Path(__file__)),
            'scope':'Local private full-coverage supplement for flagged cases outside the original stratified sample. Status and IDs are hidden, but the reviewer knows this is a targeted numeric-flag packet. Pattern flags are not errors; no human judgments are supplied. This nonrandom supplement cannot be pooled into population estimates without a revised design.'}
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    counts=check(output/'blind-review.jsonl',output/'blind-review.jsonl',report)
    print(json.dumps({'cases':report['cases'],'flagged_population':report['flagged_population'],'already_in_original_sample':report['already_in_original_sample'],'review_counts':counts,'scope':report['scope']},indent=2))


if __name__=='__main__':main()
