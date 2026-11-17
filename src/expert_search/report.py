"""Generate a comparison restricted to common queries; include failures as zero."""
import json
import random
from pathlib import Path
from .repository_text import write_english_report


def generate(baseline='artifacts/benchmark', agent='artifacts/agent-benchmark', output='docs/benchmark-results.md'):
    base=Path(baseline); model=Path(agent)
    manifest=json.loads((base/'manifest.json').read_text())
    rows=[json.loads(x) for x in (base/'runs.jsonl').read_text().splitlines()]
    agent_rows=[json.loads(p.read_text()) for p in model.glob('paper-*.json')]
    common={x['query_id'] for x in agent_rows}
    lines=['# Benchmark Results','',
        'Task: Held-out paper title → known-affiliated expert recovery. Agent labels come from paper affiliations, not human judgments of relevance.',
        f"Corpus SHA256: `{manifest['corpus_sha256']}`; random seed {manifest['seed']}.",'',
        '## All baseline queries','', '| Method | Query count | nDCG@10 | MRR@10 | Recall@10 | Mean seconds |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    def table(data):
        for method in sorted({x['method'] for x in data}):
            rs=[x for x in data if x['method']==method]; n=len(rs)
            means=[sum(x['metrics'][k] for x in rs)/n for k in ['ndcg@10','mrr@10','recall@10']]
            lines.append(f'| {method} | {n} | '+ ' | '.join(f'{v:.4f}' for v in means)+f" | {sum(x['seconds'] for x in rs)/n:.3f} |")
    table(rows)
    if common:
        lines.extend(['','## Subset of queries evaluated on the same model','',
            '| Method | Query count | nDCG@10 | MRR@10 | Recall@10 | Mean seconds |',
            '| --- | ---: | ---: | ---: | ---: | ---: |'])
        normalized=[dict(x,method='hybrid-evidence-tree') for x in agent_rows]
        table([x for x in rows if x['query_id'] in common]+normalized)
        lines.extend(['',f"Model experiment failures: {sum(x['status']!='ok' for x in agent_rows)}. Failures count as zero; do not aggregate successful requests alone."])
        if len(common)>=2:
            lines.extend(['','Paired bootstrap (2,000 iterations, seed=42): difference in nDCG@10 between tree agent and BM25-vote.'])
            a={r['query_id']:r['metrics']['ndcg@10'] for r in agent_rows}
            b={r['query_id']:r['metrics']['ndcg@10'] for r in rows if r['method']=='bm25-vote'}
            differences=[a[q]-b[q] for q in sorted(common)]
            rng=random.Random(42)
            estimates=sorted(sum(rng.choices(differences,k=len(differences)))/len(differences) for _ in range(2000))
            lines.append(f'Mean difference {sum(differences)/len(differences):.4f}, 95% bootstrap interval [{estimates[49]:.4f}, {estimates[1949]:.4f}]. Results from a small sample should not be extrapolated.')
        successful=[x for x in agent_rows if x['status']=='ok']
        if successful:
            prompt=sum(x['response']['usage']['prompt_tokens'] for x in successful)
            completion=sum(x['response']['usage']['completion_tokens'] for x in successful)
            recall=sum(len(set(x['response']['candidate_ids'])&{e for e,g in x['qrels'].items() if g>0})/sum(g>0 for g in x['qrels'].values()) for x in successful)/len(successful)
            lines.extend(['',f'Total prompt tokens for successful requests: {prompt}; completion tokens: {completion}. No price schedule is configured, so cost is not estimated.',f'Known-author recall in candidate sets for successful requests: {recall:.4f}; this constrains the upper bound achievable by subsequent ranking.'])
    lines.extend(['','## Limitations on interpretation','',
        'Author affiliations do not include all relevant experts; Recall measures recovery of known labels. Evidence matching the title and all fields containing that exact full title are excluded; related papers and paraphrased content may still provide information. Set statistics still include excluded documents.',
        'Baselines run sequentially in the same process, so timings include cache effects; model timings are also affected by service load. Without human gold labels and evidence of statistical significance, these results cannot support claims of superiority on real enterprise needs.',
        'The tree agent’s first-stage candidates are constrained by BM25; subsequent model verification may improve explanations or may hurt ranking. Inspect per-query results as well.',
        'Raw data: artifacts/benchmark/runs.jsonl; model-stage outputs and sources: artifacts/agent-benchmark/paper-*.json.'])
    write_english_report(output,'\n'.join(lines)+'\n')
    return output
