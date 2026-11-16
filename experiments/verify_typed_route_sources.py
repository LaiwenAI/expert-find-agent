"""Source-reading follow-up to cached typed routes; no silver labels in prompts."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

from expert_search.applicability import rank_candidates
from expert_search.expert_catalog import ExpertCatalog
from expert_search.source_applicability import SourceApplicability, PROMPT, resolve_assessments


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--span-ids", action="store_true")
    args = parser.parse_args()
    if args.span_ids:
        from expert_search.span_applicability import SpanApplicability, PROMPT as span_prompt, resolve_span_assessments, source_spans
        assessor_class, active_prompt = SpanApplicability, span_prompt
        parser_source = inspect.getsource(resolve_span_assessments)+inspect.getsource(source_spans)
    else:
        assessor_class, active_prompt = SourceApplicability, PROMPT
        parser_source = inspect.getsource(resolve_assessments)
    trees = [json.loads(p.read_text()) for p in sorted(Path('artifacts/expert-semantic-tree-v1').glob('CN_*.json'))]
    mapping = {}
    for tree in trees:
        audit = json.loads((Path('artifacts/expert-tree-node-audit-v1')/(tree['expert_id']+'.json')).read_text())
        supported = {j['id'] for j in audit.get('response', {}).get('judgments', []) if j['label']=='supported'}
        for node in tree.get('nodes', []):
            if node['id'] in supported:
                mapping['n'+str(len(mapping))] = dict(expert_id=tree['expert_id'], node=node)
    jobs = {j['id']: j for j in map(json.loads, Path('artifacts/paper-backed-overlap-v1/jobs.jsonl').read_text().splitlines())}
    out = Path('artifacts/span-applicability-route-v1' if args.span_ids else 'artifacts/source-applicability-route-v1'); out.mkdir(exist_ok=True)
    catalog = ExpertCatalog(); assessor = assessor_class(); tasks = []
    for path in sorted(Path('artifacts/typed-expert-tree-route-v1').glob('demand-*.json')):
        route = json.loads(path.read_text())
        if route['status'] != 'valid': continue
        key = route['id']
        plan = json.loads((Path('artifacts/routing-requirements-v1')/(key+'.json')).read_text())
        if plan['status'] != 'structurally_valid': raise ValueError('Unresolved requirement plan')
        requirements = {r['id']: dict(text=r['text'], citations=r['citations']) for r in plan['requirements']}
        candidates = route['ranking'][:3]
        owned = {s['id']: s for s in catalog.for_experts(candidates)}
        ordered_nodes = sorted(route['response']['judgments'], key=lambda j: (-j['routing_score'], j['id']))
        for eid in candidates:
            ids = []
            for judgment in ordered_nodes:
                item = mapping[judgment['id']]
                if item['expert_id'] != eid: continue
                for citation in item['node']['citations']:
                    if citation['evidence_id'] not in ids: ids.append(citation['evidence_id'])
            sources = {}
            for sid in ids[:8]:
                s = owned[sid]
                sources[sid] = dict(id=sid, expert_id=eid, text=s['text'][:1200])
            tasks.append(dict(query_id=key, expert_id=eid, demand=jobs[key]['demand'], requirements=requirements, sources=sources))
    protocol = dict(version=1, prompt=active_prompt, parser_sha256=hashlib.sha256(parser_source.encode()).hexdigest(),
                    inputs_sha256=hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest(),
                    selection='Top three experts from cached 70-node typed router; up to eight owned node-linked sources, 1200 characters each. No known-positive injection.',
                    scope='Two reused development demands and a 21-expert directory. Model applicability judgments are not evaluation gold. Planning/routing cached; no end-to-end latency claim.')
    marker = out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text()) != protocol: raise ValueError('Changed source applicability inputs')
    marker.write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    rows = []
    for task in tasks:
        target = out/(task['query_id']+'--'+task['expert_id']+'.json')
        if target.exists(): row = json.loads(target.read_text())
        else:
            row = dict(query_id=task['query_id'], expert_id=task['expert_id'], input=task, status='unresolved')
            try: row.update(assessor.assess(task['demand'], task['expert_id'], task['requirements'], task['sources']))
            except Exception as error: row.update(error_type=type(error).__name__, http_status=getattr(error, 'status_code', None))
            temporary = target.with_suffix('.partial'); temporary.write_text(json.dumps(row, ensure_ascii=False, indent=2)); temporary.replace(target)
        rows.append(row); print(task['query_id'], task['expert_id'], row['status'], flush=True)
    rankings = {}
    for qid in sorted({r['query_id'] for r in rows}):
        rankings[qid] = rank_candidates([r['result'] for r in rows if r['query_id']==qid and r['status']=='structurally_valid'])
    report = dict(candidates=len(rows), structurally_valid=sum(r['status']=='structurally_valid' for r in rows),
                  source_reading_request_seconds=sum(r.get('seconds', 0) for r in rows), provisional_rankings=rankings,
                  notice='Exact quotes and ownership checked. Semantic applicability requires separate review. Unresolved responses are not negative expert judgments.')
    (out/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='provisional_rankings'}, indent=2))


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv('.env'); run()
