"""Analyze one validated stratified review without imputing missing judgments."""
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .check_grounding_review import FIELDS, check, read


def intervals(counts, sizes):
    """Return weighted sample intervals and finite-population worst-case bounds."""
    total=sum(sizes.values()); weighted_low=weighted_high=known_positive=known_negative=0
    for key,size in sizes.items():
        c=counts[key]; n=sum(c.values())
        if not n:raise ValueError('Empty sample stratum')
        positive=c['supported'];negative=c['changed_or_unsupported']
        weighted_low+=size*positive/n;weighted_high+=size*(n-negative)/n
        known_positive+=positive;known_negative+=negative
    return {'weighted_sample_interval':[weighted_low/total,weighted_high/total],
            'finite_population_worst_case':[known_positive/total,(total-known_negative)/total]}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('review',type=Path)
    parser.add_argument('--root',type=Path,default=Path('artifacts/stratified-grounding-human-review-v1'))
    args=parser.parse_args();root=args.root;data=Path('artifacts/dataset-release/bilingual-query-construction-v1')
    manifest=json.loads((root/'manifest.json').read_text()); dataset=json.loads((data/'manifest.json').read_text())
    if hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest()!=manifest['dataset_manifest_sha256']:raise ValueError('Changed dataset manifest')
    for name,digest in dataset['outputs_sha256'].items():
        if hashlib.sha256((data/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed dataset output')
    baseline=root/'blind-review.jsonl';check(baseline,args.review,manifest)
    reviewed=read(args.review);original=read(baseline)
    mapping=json.loads((root/'private-mapping.json').read_text())
    mapping_hash=hashlib.sha256((root/'private-mapping.json').read_bytes()).hexdigest()
    if 'private_mapping_sha256' in manifest and mapping_hash!=manifest['private_mapping_sha256']:raise ValueError('Changed mapping')
    strata=defaultdict(list)
    for row in map(json.loads,(data/'synthetic-queries.jsonl').read_text().splitlines()):strata[row['intent_type'],row['status']].append(row)
    rng=random.Random(manifest['seed']);selected=[];sizes={}
    for spec in manifest['strata']:
        key=(spec['intent_type'],spec['status']);rows=sorted(strata[key],key=lambda r:r['id'])
        if key in sizes or len(rows)!=spec['population']:raise ValueError('Stratum population mismatch')
        sizes[key]=len(rows);selected.extend(rng.sample(rows,spec['sample']))
    if set(sizes)!=set(strata):raise ValueError('Missing strata')
    rng.shuffle(selected)
    expected=[{'case_id':f'case-{i:03d}','query_id':r['id'],'parent_id':r['parent_id'],'status':r['status']} for i,r in enumerate(selected,1)]
    if mapping!=expected:raise ValueError('Mapping differs from deterministic sample')
    parents={r['id']:r for r in map(json.loads,(data/'real-queries.jsonl').read_text().splitlines())}
    counts={field:defaultdict(Counter) for field in FIELDS}
    for item,row in zip(mapping,selected):
        source=original[item['case_id']];judge=reviewed[item['case_id']]
        if (source['intent_type'],source['original_need'],source['chinese_request'],source['english_request'])!=(row['intent_type'],parents[row['parent_id']]['zh'],row['zh'],row['en']):raise ValueError('Review source mismatch')
        for field in FIELDS:counts[field][row['intent_type'],row['status']][judge[field] or 'missing']+=1
    results={field:{**intervals(counts[field],sizes),'counts':dict(sum(counts[field].values(),Counter()))} for field in FIELDS}
    print(json.dumps({'fields':results,'cases':len(mapping),'private_mapping_sha256':mapping_hash,
                      'review_sha256':hashlib.sha256(args.review.read_bytes()).hexdigest(),
                      'scope':'Weighted sample intervals reflect missing and uncertain labels only, not sampling uncertainty. Finite-population worst-case bounds leave all unsampled pairs unknown; neither interval is a confidence interval. One reviewer, construction split only.'},indent=2))


if __name__=='__main__':main()
