"""Sample comparison-relevant fixed views for a blinded development review."""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(line) for line in p.read_text().splitlines()]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('packet',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=Path('artifacts/dataset-release/fresh-enterprise-eval-v1');audit=Path('artifacts/fresh-lexical-pool-audit-v1/runs.jsonl')
    manifest=json.loads((args.packet/'manifest.json').read_text())
    for name in ('blind-review.jsonl','private-mapping.jsonl'):
        if sha(args.packet/name)!=manifest['outputs_sha256'][name]:raise ValueError('Changed packet')
    if sha(root/'pools.jsonl')!=manifest['inputs_sha256']['pools.jsonl'] or sha(audit)!=manifest['inputs_sha256']['lexical_runs']:
        raise ValueError('Changed rankings')
    forms={r['case_id']:r for r in read(args.packet/'blind-review.jsonl')};mapping=read(args.packet/'private-mapping.jsonl')
    cells={(r['query_id'],r['expert_id']):r for r in mapping}
    if len(cells)!=len(mapping) or set(forms)!={r['case_id'] for r in mapping}:raise ValueError('Invalid case mapping')
    dense={r['query_id']:r['dense_order'][:5] for r in read(root/'pools.jsonl') if r['split']=='development'}
    lex={r['query_id']:[x['expert_id'] for x in r['ranking'][:5]] for r in read(audit) if r['variant']=='full-demand'}
    if set(dense)!=set(lex) or len(dense)!=80:raise ValueError('Changed cohort')
    rng=random.Random(20261126);chosen=[];eligible=Counter()
    for q in sorted(dense):
        a=set(dense[q]);b=set(lex[q])
        if len(a)!=5 or len(b)!=5:raise ValueError('Invalid ranking')
        for bucket,candidates in (('dense_only',sorted(a-b)),('lexical_only',sorted(b-a))):
            if not candidates:continue
            eligible[bucket]+=len(candidates);eid=rng.choice(candidates)
            if (q,eid) not in cells:raise ValueError('Missing fixed view')
            chosen.append((cells[q,eid],bucket,len(candidates)))
    rng.shuffle(chosen)
    if len({m['case_id'] for m,_,_ in chosen})!=len(chosen):raise ValueError('Repeated sample')
    selected_forms=[forms[m['case_id']] for m,_,_ in chosen]
    selected_mapping=[m for m,_,_ in chosen]
    private_design=[{'case_id':m['case_id'],'selection_bucket':bucket,'probability':1/size,'stratum_size':size} for m,bucket,size in chosen]
    args.output.mkdir(parents=True,exist_ok=False)
    for name,rows in [('blind-review.jsonl',selected_forms),('private-mapping.jsonl',selected_mapping),('private-design.jsonl',private_design)]:
        (args.output/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    report={'candidate_views':len(chosen),'queries':len(dense),'seed':20261126,'eligible_by_bucket':dict(eligible),
            'sampled_by_bucket':dict(Counter(bucket for _,bucket,_ in chosen)),
            'input_sha256':{'parent_manifest':sha(args.packet/'manifest.json'),'pools':sha(root/'pools.jsonl'),'lexical_runs':sha(audit),'script':sha(Path(__file__))},
            'outputs_sha256':{n:sha(args.output/n) for n in ('blind-review.jsonl','private-mapping.jsonl','private-design.jsonl')},
            'scope':'Private blinded development sample: at most one uniformly sampled dense-only and lexical-only view per query, selected after examining rankings. Not a representative sample of all candidates; selected method/stratum hidden in reviewer forms but source content may identify it. Prior labels do not transfer. No human judgments. Known per-query stratum probabilities do not establish independence or remove nonresponse bias.'}
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
