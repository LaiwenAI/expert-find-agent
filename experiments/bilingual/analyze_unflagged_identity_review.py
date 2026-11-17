"""Validate a blinded identity review and bound unresolved sample outcomes."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path


def weighted_intervals(counts, sizes):
    """Return stratified sample bounds and worst-case finite-population bounds."""
    total=sum(sizes.values());low=high=known_yes=known_no=0
    if total<=0 or set(counts)!=set(sizes):raise ValueError('Invalid strata')
    for key,population in sizes.items():
        count=counts[key];n=sum(count.values())
        if (type(population)!=int or population<1 or not n or n>population or set(count)-{'yes','no','uncertain','missing'}
                or any(type(v)!=int or v<0 for v in count.values())):raise ValueError('Invalid stratum counts')
        yes=count.get('yes',0);no=count.get('no',0);low+=population*yes/n;high+=population*(n-no)/n
        known_yes+=yes;known_no+=no
    return dict(weighted_sample_bounds=[low/total,high/total],
                finite_population_worst_case=[known_yes/total,(total-known_no)/total])


def validate_rows(original, reviewed):
    """Check immutable input fields and permit explicitly partial human labels."""
    baseline={r['case_id']:r for r in original};values={r['case_id']:r for r in reviewed}
    if len(baseline)!=len(original) or len(values)!=len(reviewed) or set(baseline)!=set(values):raise ValueError('Missing or duplicate cases')
    immutable=('case_id','english_request')
    fields=('specific_identity','indirect_identifiers')
    for case,row in values.items():
        base=baseline[case]
        if set(row)!=set(base) or any(row[key]!=base[key] for key in immutable):raise ValueError('Changed review input')
        if any(row[field] not in (None,'yes','no','uncertain') for field in fields):raise ValueError('Invalid review label')
        if row['notes'] is not None and not isinstance(row['notes'],str):raise ValueError('Invalid notes')
        if row['reviewer_id'] is not None and (not isinstance(row['reviewer_id'],str) or not row['reviewer_id'].strip()):raise ValueError('Invalid reviewer')
        if any(row[f] is not None for f in fields) and not row['reviewer_id']:raise ValueError('Reviewer required for a judgment')
    return values


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('review',type=Path);args=parser.parse_args()
    root=Path('artifacts/unflagged-identity-human-review-v1');audit=Path('artifacts/unflagged-identity-sample-v1');data=Path('artifacts/dataset-release/english-alignment-gated-v1')
    manifest=json.loads((root/'manifest.json').read_text());protocol=json.loads((audit/'protocol.json').read_text());dataset=json.loads((data/'manifest.json').read_text())
    sha=lambda b:hashlib.sha256(b).hexdigest()
    form=(root/'blind-review.jsonl').read_bytes();mapping_raw=(root/'private-mapping.json').read_bytes();source=(data/'queries.jsonl').read_bytes()
    if (sha(form)!=manifest['review_sha256'] or sha(mapping_raw)!=manifest['mapping_sha256'] or sha(source)!=manifest['source_sha256'] or
        sha(source)!=dataset['output_sha256'] or sha((audit/'protocol.json').read_bytes())!=manifest['audit_protocol_sha256']):raise ValueError('Changed review source')
    baseline=[json.loads(line) for line in form.splitlines()];reviewed=[json.loads(line) for line in args.review.read_bytes().splitlines()]
    reviews=validate_rows(baseline,reviewed);mapping=json.loads(mapping_raw);sources={r['id']:r for r in map(json.loads,source.splitlines())}
    if len(mapping)!=200 or {m['case_id'] for m in mapping}!=set(reviews) or len({m['id'] for m in mapping})!=200 or set(m['id'] for m in mapping)!=set(protocol['selected_ids']):raise ValueError('Invalid sample mapping')
    sizes={(r['intent_type'],r['status']):r['population'] for r in protocol['strata']}
    if len(sizes)!=10:raise ValueError('Invalid design')
    counts={field:defaultdict(Counter) for field in ('specific_identity','indirect_identifiers')};confusion=Counter();baselines={r['case_id']:r for r in baseline}
    for item in mapping:
        row=sources[item['id']];key=(item['intent_type'],item['status']);review=reviews[item['case_id']]
        if key not in sizes or (row['intent_type'],row['status'])!=key or row['english']!=baselines[item['case_id']]['english_request'] or row['deidentification']!='not_assessed':raise ValueError('Source or stratum mismatch')
        path=audit/'records'/(hashlib.sha256(item['id'].encode()).hexdigest()+'.json');body=path.read_bytes()
        if sha(body)!=manifest['audit_record_sha256'][item['id']]:raise ValueError('Changed audit record')
        model=json.loads(body)
        if model['id']!=item['id'] or model['status']!=item['model_status'] or model.get('label')!=item['model_label']:raise ValueError('Changed model mapping')
        for field in counts:counts[field][key][review[field] or 'missing']+=1
        if model['status']=='valid' and review['specific_identity'] in ('yes','no'):
            confusion[review['specific_identity'],model['label']]+=1
    for field in counts:
        if set(counts[field])!=set(sizes) or any(sum(c.values())!=20 for c in counts[field].values()):raise ValueError('Incomplete stratum accounting')
    report=dict(sample_cases=len(mapping),fields={f:{**weighted_intervals(counts[f],sizes),'counts':dict(sum(counts[f].values(),Counter()))} for f in counts},
                model_human_common=[dict(human=a,model=b,count=n) for (a,b),n in sorted(confusion.items())],
                review_sha256=sha(args.review.read_bytes()),scope='Bounds represent missing/uncertain sample labels or unobserved population members, not sampling confidence intervals. Human review may err; stratified sample is of 4,979 rule-unflagged rows only. Repeated parents and model false negatives remain possible.')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
