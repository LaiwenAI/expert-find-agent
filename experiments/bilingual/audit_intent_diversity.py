"""Measure within-parent lexical overlap of synthetic bilingual intents."""
import argparse
from collections import Counter,defaultdict
import hashlib
import itertools
import json
import statistics
import unicodedata
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shingles(text,width=4):
    if not isinstance(text,str):raise ValueError('Require a text view')
    normalized=''.join(unicodedata.normalize('NFKC',text).casefold().split())
    return {normalized[i:i+width] for i in range(len(normalized)-width+1)}


def overlap(a,b):
    if not a or not b:return None
    return len(a&b)/len(a|b)


def main(root,output):
    manifest=json.loads((root/'manifest.json').read_text());path=root/'synthetic-queries.jsonl'
    if sha(path)!=manifest['outputs_sha256'][path.name]:raise ValueError('Changed synthetic data')
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    if len(rows)!=manifest['synthetic_slots'] or len({r['id'] for r in rows})!=len(rows):raise ValueError('Changed slot inventory')
    types=set(manifest['taxonomy']);groups=defaultdict(list)
    for row in rows:
        if row['split']!='construction' or row['id']!=row['parent_id']+':'+row['intent_type'] or row['intent_type'] not in types:
            raise ValueError('Changed parent or intent binding')
        groups[row['parent_id']].append(row)
    if len(groups)!=manifest['real_original_queries'] or any(len(rs)!=len(types) or {r['intent_type'] for r in rs}!=types for rs in groups.values()):
        raise ValueError('Incomplete parent intent groups')
    cases=[];exact={}
    for lang in ('zh','en'):
        text_groups=defaultdict(list)
        for row in rows:
            if not isinstance(row[lang],str) or not row[lang].strip():raise ValueError('Missing language view')
            normalized=' '.join(unicodedata.normalize('NFKC',row[lang]).casefold().split())
            text_groups[normalized].append(row['id'])
        exact[lang]=dict(distinct_texts=len(text_groups),duplicate_groups=sum(len(x)>1 for x in text_groups.values()),duplicate_rows=sum(len(x) for x in text_groups.values() if len(x)>1))
    for parent,rs in sorted(groups.items()):
        for a,b in itertools.combinations(sorted(rs,key=lambda x:x['intent_type']),2):
            values={lang:overlap(shingles(a[lang]),shingles(b[lang])) for lang in ('zh','en')}
            cases.append(dict(parent_id=parent,intent_pair=[a['intent_type'],b['intent_type']],ids=[a['id'],b['id']],statuses=[a['status'],b['status']],overlap=values))
    def aggregate(items,lang):
        values=[r['overlap'][lang] for r in items if r['overlap'][lang] is not None]
        return dict(pairs=len(items),scorable=len(values),unscorable=len(items)-len(values),median=statistics.median(values) if values else None,
            at_least_05=sum(v>=.5 for v in values),at_least_08=sum(v>=.8 for v in values),maximum=max(values,default=None))
    intent_pairs=sorted({tuple(r['intent_pair']) for r in cases})
    results={lang:dict(overall=aggregate(cases,lang),by_intent_pair={' / '.join(k):aggregate([r for r in cases if tuple(r['intent_pair'])==k],lang) for k in intent_pairs},
                       flagged_parents_05=len({r['parent_id'] for r in cases if r['overlap'][lang] is not None and r['overlap'][lang]>=.5}),
                       retained_retained_flagged_05=sum(r['overlap'][lang] is not None and r['overlap'][lang]>=.5 and all(s=='retained_model_screened_pair' for s in r['statuses']) for r in cases)) for lang in ('zh','en')}
    output.mkdir(parents=True,exist_ok=False);(output/'private-cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    report=dict(parents=len(groups),synthetic_views=len(rows),within_parent_pairs=len(cases),exact_normalized=exact,lexical_4gram=results,
        input_sha256=sha(path),manifest_sha256=sha(root/'manifest.json'),script_sha256=sha(__file__),private_cases_sha256=sha(output/'private-cases.json'),
        scope='Within-parent lexical distinctness only, not semantic diversity, grounding, independent intent labels or expert-fit quality. Thresholds 0.5 and 0.8 are descriptive and selected after inspecting data. Jaccard depends on length and language; do not compare languages as quality scores. Each parent contributes dependent pairs. Private IDs and source texts are excluded from aggregate.')
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('artifacts/dataset-release/bilingual-query-construction-v1'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.root,a.output)
