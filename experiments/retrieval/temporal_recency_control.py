"""Compare fixed recency decay against max scoring on a historical snapshot."""
import argparse
from datetime import date, timedelta
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.metadata.papers import availability, publication_interval


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output):
    parent = Path('artifacts/local-temporal-author-proxy-v1'); history = Path('artifacts/corpus-asof-2020-v1'); previous = Path('artifacts/temporal-vote-vs-max-v1')
    meta = json.loads((parent / 'summary.json').read_text()); prior = json.loads((previous / 'summary.json').read_text())
    if sha(parent / 'results.jsonl') != meta['results_sha256'] or sha(history / 'evidence.jsonl') != meta['historical_sha256'] or sha(previous / 'results.json') != prior['results_sha256']:
        raise ValueError('Changed frozen data')
    cutoff = date(2020, 12, 31); ages = {}
    for row in map(json.loads, (history / 'evidence.jsonl').read_text().splitlines()):
        interval = publication_interval(row['publication']['raw'])
        if interval != row['publication'] or availability(interval, cutoff.isoformat()) != 'before_or_on_cutoff': raise ValueError('Unsafe or changed date')
        latest = date.fromisoformat(interval['end_exclusive']) - timedelta(days=1)
        ages[row['id']] = (cutoff - latest).days / 365.2425
    cases = [json.loads(line) for line in (parent / 'results.jsonl').read_text().splitlines()]; old = json.loads((previous / 'results.json').read_text())
    if len(cases) != 200 or [r['id'] for r in cases] != [r['id'] for r in old]: raise ValueError('Changed query cohort')
    engine = CandidateEvidencePipeline(history, 'artifacts/temporal-candidate-lm-2020-v1', 'artifacts/temporal-evidence-2020-v1.sqlite')
    engine.evidence.prepare_adaptive_scoring(); rows = []
    for case, previous_row in zip(cases, old):
        start = time.perf_counter(); pool = [r['expert']['id'] for r in engine.ranker.search(case['title'], 50)['results']]
        source_ids = [sid for eid in pool for sid in engine.evidence_by_expert[eid]]
        scores = engine.evidence.adaptive_document_scores(case['title'], source_ids)['scores']; arms = {}; truth = set(case['known_author_ids'])
        for name, half_life in (('max', None), ('half5', 5), ('half10', 10)):
            values = {}; best_age = {}
            for eid in pool:
                sources = engine.evidence_by_expert[eid]
                weighted = [(scores[sid] * (2 ** (-ages[sid] / half_life) if half_life else 1.), sid) for sid in sources]
                if weighted:
                    score, sid = min(weighted, key=lambda x: (-x[0], x[1])); values[eid] = score; best_age[eid] = ages[sid]
                else: values[eid] = 0.; best_age[eid] = None
            ranking = sorted(pool, key=lambda eid: (-values[eid], eid))[:10]
            if name == 'max' and ranking != previous_row['arms']['max']['ranking']: raise ValueError('Changed max ranking')
            ranks = [i for i,eid in enumerate(ranking,1) if eid in truth]
            arms[name] = {'ranking': ranking, 'hit': int(bool(ranks)), 'mrr': 1/min(ranks) if ranks else 0.,
                          'known_recall': len(set(ranking)&truth)/len(truth),
                          'mean_selected_source_age_years': statistics.mean(best_age[eid] for eid in ranking if best_age[eid] is not None)}
        rows.append({'id': case['id'], 'arms': arms, 'shared_seconds': time.perf_counter()-start})
    engine.evidence.db.close()
    report = {'queries': len(rows), 'cutoff': cutoff.isoformat(), 'max_rankings_match_prior': len(rows),
              'methods': {arm: {key: statistics.mean(r['arms'][arm][key] for r in rows) for key in ('hit','mrr','known_recall','mean_selected_source_age_years')} for arm in ('max','half5','half10')},
              'mean_shared_all_arms_seconds': statistics.mean(r['shared_seconds'] for r in rows),
              'input_sha256': {'parent': meta['results_sha256'], 'history': meta['historical_sha256'], 'prior_results': prior['results_sha256']},
              'scope': 'Post-hoc fixed half-life decay (5 and 10 years) on the same candidate-50 pool, using the latest possible publication date in a fixed 2020 snapshot; no cutoff-crossing or unknown dates included. Shared retrieval and scoring timing is not per-method latency. Publication date may not equal availability and source association is not individual contribution. Reused author-recovery labels are incomplete and unverified; no enterprise fit, causal or held-out policy claim.'}
    output.mkdir(parents=True, exist_ok=False); (output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    report['results_sha256'] = sha(output / 'results.json'); report['code_sha256'] = {str(p): sha(p) for p in (Path(__file__), Path('src/expert_search/metadata/papers.py'), Path('src/expert_search/candidate_evidence_pipeline.py'))}
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True); main(p.parse_args().output)
