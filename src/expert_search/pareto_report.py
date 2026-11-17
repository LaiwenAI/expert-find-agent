"""Separate comparable timing boundaries instead of mixing models and stages."""
import json
from pathlib import Path
from .repository_text import write_english_report


def generate(output='artifacts/reports/quality-time.md'):
    lines=['# Quality–time evidence table','','Separate tables for different tasks, models, and timing scopes; do not treat local computation speed as agent end-to-end speed.','',
        '## Same encoder, local retrieval stage','','|Path|SciFact nDCG@10|Mean ms|p95 ms|','|---|---:|---:|---:|']
    p=Path('docs/exploration/progressive-latency.json')
    if p.exists():
        r=json.loads(p.read_text())
        for name in ['full','progressive']:
            t=r['timings'][name];m=r['quality'][name]
            lines.append(f"|{name}|{m['ndcg@10']:.4f}|{t['mean_seconds']*1000:.3f}|{t['p95_seconds']*1000:.3f}|")
    lines+=['','These use precomputed query vectors and do not represent online user latency.','',
        '## Query encoding probe','','|Path|Samples|p50 ms|p95 ms|Quality boundary|','|---|---:|---:|---:|---|']
    p=Path('docs/exploration/local-query-latency.json')
    if p.exists():
        r=json.loads(p.read_text());lines.append(f"|Local frozen BGE|{r['samples']}|{r['p50_seconds']*1000:.2f}|{r['p95_seconds']*1000:.2f}|Full precision measured separately|")
    lines+=['','API encoding for five samples takes about 0.52–1.40 seconds; these five samples cannot be compared with twenty local samples to infer a strict p95 speedup. The models differ, so their precision results cannot be combined.','',
        '## Complete local pipeline','']
    p=Path('artifacts/public/scifact-local-bge/summary.json')
    if p.exists():
        r=json.loads(p.read_text());lines.append(f"300 queries: nDCG@10={r['metrics']['ndcg@10']:.4f}, encoding + retrieval p50={r['p50_seconds']*1000:.2f}ms, p95={r['p95_seconds']*1000:.2f}ms.")
    else:lines.append('Complete encoding/evaluation artifacts are unavailable; no full-pipeline quality–time result is reported.')
    lines+=['','## Inference paths','','CMU and SciFact are different tasks and should not be compared for cross-task Pareto dominance. Online verification includes model generation and must report accuracy, request count, offline cost, and model latency alongside the original-profile baseline.']
    write_english_report(output,'\n'.join(lines)+'\n');return output

if __name__=='__main__':print(generate())
