"""Decompose bilingual paper-winner differences into common and unique experts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
from expert_search.statistics import clustered_paired_interval


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def decompose(zh,en):
    """Decompose one pair's top-ten slot fractions without assessing relevance."""
    if any(len(v['ranking'])!=10 or len(set(v['ranking']))!=10 or len(v['source_kinds'])!=10 or any(k not in ('paper','profile') for k in v['source_kinds']) for v in (zh,en)):
        raise ValueError('Invalid rank or source-kind list')
    z=dict(zip(zh['ranking'],zh['source_kinds']));e=dict(zip(en['ranking'],en['source_kinds']));shared=z.keys()&e.keys()
    transitions=Counter(z[k]+'->'+e[k] for k in shared)
    common=(sum(e[k]=='paper' for k in shared)-sum(z[k]=='paper' for k in shared))/10
    unique=(sum(e[k]=='paper' for k in e.keys()-shared)-sum(z[k]=='paper' for k in z.keys()-shared))/10
    if abs(common+unique-(sum(x=='paper' for x in e.values())-sum(x=='paper' for x in z.values()))/10)>1e-12:raise ValueError('Decomposition failed')
    return dict(common_experts=len(shared),transitions=dict(transitions),common_contribution=common,unique_contribution=unique,
                zh_unique_paper=sum(z[k]=='paper' for k in z.keys()-shared),en_unique_paper=sum(e[k]=='paper' for k in e.keys()-shared),unique_experts=10-len(shared))


def main(output):
    root=Path('artifacts/bilingual-all-source-diagnostic-v2');summary=json.loads((root/'summary.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    if sha(root/'protocol.json')!=summary['protocol_sha256'] or sha(root/'private-results.json')!=summary['private_results_sha256']:raise ValueError('Changed source diagnostics')
    prior=Path('artifacts/bilingual-intent-retrieval-results-v1')
    if sha(prior/'summary.json')!=protocol['prior_summary_sha256'] or sha(prior/'private-results.json')!=protocol['prior_results_sha256']:raise ValueError('Changed retrieval inputs')
    rows=json.loads((root/'private-results.json').read_text());previous=json.loads((prior/'private-results.json').read_text())
    if len(rows)!=5000 or len(previous)!=5000 or [r['id'] for r in rows]!=[r['id'] for r in previous]:raise ValueError('Changed query inventory')
    cases=[];transitions=Counter()
    for r,p in zip(rows,previous):
        if any(r[k]!=p[k] for k in ('id','parent_id','group_id','intent_type','status')) or any(r['views'][l]['ranking']!=p['ranks'][l] for l in ('zh','en')):raise ValueError('Changed query binding or ranks')
        value=decompose(r['views']['zh'],r['views']['en']);transitions.update(value['transitions']);cases.append(dict(group_id=r['group_id'],intent_type=r['intent_type'],**value))
    common=sum(r['common_experts'] for r in cases);unique=sum(r['unique_experts'] for r in cases)
    if common==0 or unique==0:raise ValueError('No common or unique experts')
    def interval(field):
        x=clustered_paired_interval([r[field] for r in cases],[r['group_id'] for r in cases]);x['sampling_unit']='whole company group; query-weighted mean';return x
    common_mean=statistics.mean(r['common_contribution'] for r in cases);unique_mean=statistics.mean(r['unique_contribution'] for r in cases)
    report=dict(queries=len(cases),shared_expert_slots=common,unique_slots_per_language=unique,shared_kind_transitions=dict(transitions),
        shared_kind_switch_fraction=(transitions['paper->profile']+transitions['profile->paper'])/common,
        zh_unique_paper_fraction=sum(r['zh_unique_paper'] for r in cases)/unique,en_unique_paper_fraction=sum(r['en_unique_paper'] for r in cases)/unique,
        mean_english_minus_chinese_paper_fraction=common_mean+unique_mean,shared_contribution=interval('common_contribution'),unique_contribution=interval('unique_contribution'),
        input_sha256={str(p):sha(p) for p in (root/'protocol.json',root/'summary.json',root/'private-results.json',prior/'summary.json',prior/'private-results.json')},script_sha256=sha(__file__),
        scope='Algebraic post-hoc accounting of displayed top-ten source types; shared and unique membership are outputs of the same ranker, not randomized interventions or causes. Source-kind tags do not establish evidence relevance, authorship, or contribution. Shared slots are repeated query-expert pairs; company-group bootstrap is query-weighted and unadjusted. Reused model-screened data are not held-out labels.')
    output.mkdir(parents=True,exist_ok=False);(output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
