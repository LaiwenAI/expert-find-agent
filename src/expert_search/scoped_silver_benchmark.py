"""Source-state diagnostics on audited silver pairs, with full input reconstruction.

This is not expert retrieval quality: other experts are unjudged. The literal
baseline receives the generated condition structure, so it excludes parsing.
"""
import hashlib
import json
from pathlib import Path
from .corpus import records


def load(directory='artifacts/dataset-release/scoped-capabilities-audited',split='test',corpus='artifacts/corpus'):
    root=Path(directory)
    pairs=root/'pairs.jsonl'
    if split not in ('all','train','validation','test'):
        raise ValueError('Unknown split')
    if split=='all':
        rows=[r for _,r in records(pairs)]
    else:
        info=json.loads((root/'splits.json').read_text())
        target=root/(split+'.jsonl')
        if not info.get('feasible') or hashlib.sha256(pairs.read_bytes()).hexdigest()!=info['pairs_sha256']:
            raise ValueError('Missing or stale component split')
        if hashlib.sha256(target.read_bytes()).hexdigest()!=info['file_sha256'][target.name]:
            raise ValueError('Changed split file')
        rows=[r for _,r in records(target)]
        if sorted(r['id'] for r in rows)!=info['assignments'][split]:
            raise ValueError('Split assignment mismatch')
    if not rows:
        raise ValueError('Empty benchmark')
    wanted={s['evidence_id'] for r in rows for s in r['source_snapshots']}
    sources={r['id']:r for _,r in records(Path(corpus)/'evidence.jsonl') if r['id'] in wanted}
    if wanted-set(sources):
        raise ValueError('Missing input source snapshot')
    cases=[]
    for row in rows:
        evidence=[]
        for snapshot in row['source_snapshots']:
            source=sources[snapshot['evidence_id']]
            length=snapshot['prefix_characters']
            if type(length)!=int or length<1:
                raise ValueError('Invalid source prefix length')
            text=source['text'][:length]
            if (len(text)!=length or source['expert_id']!=row['constructing_expert_id'] or
                    snapshot['expert_id']!=source['expert_id'] or snapshot['source']!=source['source'] or
                    hashlib.sha256(text.encode()).hexdigest()!=snapshot['source_snapshot_sha256']):
                raise ValueError('Changed source content or ownership')
            evidence.append(dict(id=source['id'],text=text))
        constraints=row['constraints']
        index=row['changed_index']
        if not 2<=len(constraints)<=3 or type(index)!=int or not 0<=index<len(constraints):
            raise ValueError('Invalid condition structure')
        changed=list(constraints)
        changed[index]=row['replacement_text']
        for variant,query,conditions,label in (
            ('base',row['query'],constraints,row['base_source_support']),
            ('counterfactual',row['counterfactual_query'],changed,row['counterfactual_source_support'])):
            if label not in ('supported','unknown'):
                raise ValueError('Unexpected source-state label')
            # Labels remain separate from scorer inputs.
            cases.append(dict(pair_id=row['id'],variant=variant,label=label,
                input=dict(query=query,conditions=conditions,evidence=evidence)))
    return cases


def literal_topics(payload):
    texts=[e['text'].casefold() for e in payload['evidence']]
    return 'supported' if all(any(c.casefold() in text for text in texts) for c in payload['conditions']) else 'unknown'


def evaluate(cases,predict):
    if not cases:
        raise ValueError('Empty evaluation')
    predictions=[predict(case['input']) for case in cases]
    if any(p not in ('supported','unknown') for p in predictions):
        raise ValueError('Invalid predicted source state')
    correct=[p==c['label'] for p,c in zip(predictions,cases)]
    true_supported=sum(c['label']=='supported' for c in cases)
    predicted_supported=predictions.count('supported')
    true_positive=sum(p=='supported' and c['label']=='supported' for p,c in zip(predictions,cases))
    pairs={}
    for case,hit in zip(cases,correct):
        if case['variant'] in pairs.setdefault(case['pair_id'],{}):
            raise ValueError('Duplicate paired variant')
        pairs[case['pair_id']][case['variant']]=hit
    if any(set(items)!= {'base','counterfactual'} for items in pairs.values()):
        raise ValueError('Incomplete pair')
    return dict(cases=len(cases),pairs=len(pairs),silver_state_accuracy=sum(correct)/len(cases),
        both_variants_correct=sum(all(items.values()) for items in pairs.values())/len(pairs),
        supported_precision=true_positive/predicted_supported if predicted_supported else None,
        supported_recall=true_positive/true_supported if true_supported else None,
        supported_prediction_rate=predicted_supported/len(cases))


def run(split='test'):
    cases=load(split=split)
    report=dict(split=split,results={name:evaluate(cases,predict) for name,predict in
        [('always-unknown',lambda _: 'unknown'),('literal-all-topics',literal_topics)]},
        warning='Small silver source-state diagnostic, not expert ranking or actual ability. Literal baseline receives oracle generated conditions; parsing and natural-query generalization excluded. Full original input snapshots reconstructed, not only cited positives. Substring presence is a heuristic and can miss negation or context. No-prediction precision is undefined, not perfect. No statistical superiority claim from this tiny pilot.')
    target=Path('docs/exploration')/f'scoped-silver-baselines-{split}.json'
    target.write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=['all','train','validation','test'],default='test')
    args=parser.parse_args()
    print(json.dumps(run(args.split),indent=2))
