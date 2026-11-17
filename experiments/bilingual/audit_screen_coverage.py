"""Summarize construction-screen coverage without exposing query text."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path('artifacts/dataset-release/bilingual-query-construction-v1')
    post=Path('artifacts/large-bilingual-data-v1/postvalidation-v2/pairs.jsonl')
    manifest=json.loads((root/'manifest.json').read_text())
    for name,expected in manifest['outputs_sha256'].items():
        if sha(root/name)!=expected:raise ValueError('Changed dataset output')
    if sha(post)!=manifest['input_sha256'][str(post)]:raise ValueError('Changed postchecks')
    real=[json.loads(line) for line in (root/'real-queries.jsonl').read_text().splitlines()]
    rows=[json.loads(line) for line in (root/'synthetic-queries.jsonl').read_text().splitlines()]
    checks=[json.loads(line) for line in post.read_text().splitlines()]
    parents={r['id']:r for r in real}
    index={(r['parent_id'],r['intent_type']):r for r in checks}
    if len(parents)!=len(real) or len(index)!=len(checks):raise ValueError('Duplicate parent or postcheck')
    domain_names={'':'unspecified','\u65b0\u519c\u79d1':'agriculture','\u65b0\u533b\u79d1':'medicine',
                  '\u65b0\u5de5\u79d1':'engineering','\u65b0\u6587\u79d1':'humanities'}
    by_type=defaultdict(Counter);by_category=defaultdict(Counter);cross=Counter();per_parent=Counter()
    for row in rows:
        check=index[row['parent_id'],row['intent_type']]
        if row['status']!=check['status']:raise ValueError('Export and postcheck disagree')
        status=row['status'];retained=status=='retained_model_screened_pair'
        by_type[row['intent_type']][status]+=1
        category=parents[row['parent_id']]['category']
        if category not in domain_names:raise ValueError('Unknown domain category')
        by_category[domain_names[category]][status]+=1
        cross[(check['prior_status'],check['numeric_postcheck']['status'],status)]+=1
        if retained:per_parent[row['parent_id']]+=1
    histogram=Counter(per_parent.get(r['id'],0) for r in real)
    result={'parents':len(real),'pairs':len(rows),'status_by_type':{k:dict(v) for k,v in sorted(by_type.items())},
            'status_by_domain':{str(k):dict(v) for k,v in sorted(by_category.items(),key=lambda x:str(x[0]))},
            'retained_pairs_per_parent':{str(k):v for k,v in sorted(histogram.items())},
            'screen_cross_counts':[{'prior':a,'numeric':b,'final':c,'count':n} for (a,b,c),n in sorted(cross.items(),key=lambda item:tuple(str(x) for x in item[0]))],
            'input_sha256':{'dataset_manifest':sha(root/'manifest.json'),'postchecks':sha(post)},
            'scope':'Construction split only. Retention measures same-model screening and limited numeric checks, not human quality or expert relevance. A numeric match is not semantic equivalence.'}
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
