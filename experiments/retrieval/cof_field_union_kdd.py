"""Transfer a fixed field-union shortlist to judged KDD candidates only."""
import hashlib
import json
import statistics
import time
from pathlib import Path

from expert_search.certified_topk import certified_topk
from expert_search.cof_benchmark import load
from expert_search.compiled_language_model import CompiledCandidateLM
from expert_search.expert_language_models import ExpertLanguageModels
from expert_search.statistics import paired_interval


def main(output):
    root, papers, reviewers, labels = load('KDD')
    documents = {pid: (papers[pid].get('title', '') + '\n' + papers[pid].get('abstract', '')).strip()
                 for pid in sorted({pid for r in reviewers.values() for pid in r['papers']})}
    start = time.perf_counter()
    model = CompiledCandidateLM(ExpertLanguageModels(documents, {eid: r['papers'] for eid, r in reviewers.items()}))
    build_seconds = time.perf_counter() - start
    if set(labels['soft']) != set(labels['hard']) or any(set(labels['soft'][q]) != set(labels['hard'][q]) or len(labels['soft'][q]) != 20 for q in labels['soft']):
        raise ValueError('Changed candidate judgments')
    audit = json.loads(Path('docs/exploration/cof-kdd-audit.json').read_text())
    copied = set(audit['exact_query_texts_in_profiles'])
    rows = []
    for i, qid in enumerate(sorted(labels['soft'])):
        paper = papers[qid]; title = paper.get('title', '').strip(); abstract = paper.get('abstract', '').strip()
        query = (title + '\n' + abstract).strip(); eligible = sorted(labels['soft'][qid]); arms = {}
        if set(eligible) - model.available: raise ValueError('Unavailable judged candidate')
        for arm in (('full10', 'union10') if i % 2 == 0 else ('union10', 'full10')):
            start = time.perf_counter()
            full = [r['id'] for r in certified_topk(model, query, 10, eligible=eligible)['results']]
            if arm == 'union10':
                pool = []
                for text in (title, abstract):
                    if text: pool.extend(r['id'] for r in certified_topk(model, text, 5, eligible=eligible)['results'])
                pool = list(dict.fromkeys(pool + full))[:10]
            else: pool = full
            elapsed = time.perf_counter() - start
            metrics = {}
            for task in labels:
                positive = {eid for eid, score in labels[task][qid].items() if score > 0}
                metrics[task] = {'recall': len(set(pool) & positive) / len(positive) if positive else None,
                                 'oracle_p5': min(len(set(pool) & positive), 5) / 5}
            arms[arm] = {'seconds': elapsed, 'metrics': metrics}
        rows.append({'query_id': qid, 'exact_copy': qid in copied, 'arms': arms})
    cohorts = {'all': rows, 'exclude_exact_copy': [r for r in rows if not r['exact_copy']]}
    report = {'queries': len(rows), 'model_build_seconds': build_seconds, 'cohorts': {}}
    for name, subset in cohorts.items():
        result = {'queries': len(subset), 'mean_seconds': {a: statistics.mean(r['arms'][a]['seconds'] for r in subset) for a in ('full10', 'union10')}, 'metrics': {}, 'paired': {}}
        for task in labels:
            result['metrics'][task] = {}; result['paired'][task] = {}
            for metric in ('recall', 'oracle_p5'):
                valid = [r for r in subset if r['arms']['full10']['metrics'][task][metric] is not None]
                result['metrics'][task][metric] = {a: statistics.mean(r['arms'][a]['metrics'][task][metric] for r in valid) for a in ('full10', 'union10')}
                result['paired'][task][metric] = paired_interval([r['arms']['union10']['metrics'][task][metric] - r['arms']['full10']['metrics'][task][metric] for r in valid])
        report['cohorts'][name] = result
    report['input_sha256'] = {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in ('KDD_papers_test.json', 'KDD_reviewers_test.json', 'KDD_queries_test_soft.json', 'KDD_queries_test_hard.json')}
    report['audit_sha256'] = hashlib.sha256(Path('docs/exploration/cof-kdd-audit.json').read_bytes()).hexdigest()
    report['scope'] = 'Transfer diagnostic restricts search to the 20 judged reviewers supplied for each query; this candidate set uses benchmark supervision and is not deployable global retrieval. Fixed 5-title/5-abstract union with full-query fill and 10-candidate budget; no BM25 reranking or relevance claim outside the judged pool. Oracle ceilings use test labels and are not achievable quality. Query-bootstrap intervals assume independent queries; near-duplicate leakage remains possible. Timings exclude initialization, index construction, evidence and reasoning.'
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    (out / 'results.json').write_text(json.dumps(rows, indent=2) + '\n'); report['results_sha256'] = hashlib.sha256((out / 'results.json').read_bytes()).hexdigest()
    report['code_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path('src/expert_search/certified_topk.py'), Path('src/expert_search/compiled_language_model.py'))}
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
