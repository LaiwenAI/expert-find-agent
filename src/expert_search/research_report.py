"""Artifact-derived tables for the research draft; missing results stay missing."""
import json
from pathlib import Path
from .repository_text import write_english_report


def generate(output='artifacts/reports/results-generated.md'):
    lines=['# Automatically Generated Experimental Results','', 'This file reads only existing experimental artifacts; experiments without files are not reported.','']
    p=Path('artifacts/public/scifact-results/summary.json')
    if p.exists():
        r=json.loads(p.read_text());lines+=['## SciFact Lexical Baselines','','|Method|Queries|nDCG@10|MRR@10|Recall@10|p50 seconds|p95 seconds|','|---|---:|---:|---:|---:|---:|---:|']
        for name,x in r['results'].items():
            m=x['metrics'];lines.append(f"|{name}|{x['queries']}|{m['ndcg@10']:.4f}|{m['mrr@10']:.4f}|{m['recall@10']:.4f}|{x['p50_seconds']:.4f}|{x['p95_seconds']:.4f}|")
        lines+=['','Includes only local lexical queries; no model APIs.','']
    p=Path('artifacts/public/scifact-hybrid-results/summary.json')
    if p.exists():
        r=json.loads(p.read_text());lines+=['## SciFact Frozen Semantic and Hybrid Retrieval','',f"Model: {r['model']}. {r['latency_warning']}",'']
        for method,m in r['results'].items():lines.append(f"- {method}: nDCG@10={m['ndcg@10']:.4f}, Recall@10={m['recall@10']:.4f}.")
    else:lines+=['## SciFact Semantic Experiments','','Complete results have not yet been generated; partial-corpus scores are not reported.','']
    p=Path('artifacts/public/cmu-results/lexical-summary.json')
    if p.exists():
        r=json.loads(p.read_text());lines+=['## CMU Gold Single-Profile Version','',f"Lexical baseline weighted pairwise loss: {r['weighted_pairwise_loss']:.6f}, comparable pairs: {r['comparable_pairs']}.",'']
    p=Path('artifacts/public/cmu-dense-results/summary.json')
    if p.exists():
        r=json.loads(p.read_text())
        for method,m in r['results'].items():lines.append(f"- {method}: loss {m['weighted_pairwise_loss']:.6f}.")
        lines+=['',r['warning'],'']
    else:lines+=['Frozen semantic profile experiments are not yet complete.','']
    p=Path('artifacts/synthetic-evidence-first-selected/selection.json')
    if p.exists():
        r=json.loads(p.read_text());lines+=['## Synthetic Data Selection','',f"Evidence-first batch: {r['total']} attempts, {r['accepted']} retained. Retained labels remain silver; full-query and counterfactual semantics have not been sufficiently validated.",'']
    for dataset in ('sigir','nips'):
        p=Path(f'docs/exploration/compiled-lm-{dataset}.json')
        if p.exists():
            r=json.loads(p.read_text())
            lines += [f'## {dataset.upper()} Classic Candidate Language Models and Precompilation','',
                      '|Method|Soft P@5|Hard P@5|Local mean ms|','|---|---:|---:|---:|']
            for method,m in r['metrics']['soft'].items():
                lines.append(f"|{method}|{m['p@5']:.4f}|{r['metrics']['hard'][method]['p@5']:.4f}|{1000*r['timings'][method]['mean']:.3f}|")
            lines += ['',f"{r['queries']} queries, three runs per item in alternating random order; scores are identical before and after compilation. Quality belongs to the classic model; compilation changes only execution.",r['warning'],'']
        p=Path(f'docs/exploration/cof-{dataset}-packing-shared-analysis.json')
        if p.exists():
            r=json.loads(p.read_text())
            if r.get('complete'):
                lines += [f'## {dataset.upper()} Shared Evidence Across the Same Candidates Experiment','',
                          '|Evidence|Soft P@5|Hard P@5|Mean call seconds|P95 seconds|Mean input tokens|Failures|','|---|---:|---:|---:|---:|---:|---:|']
                for condition,row in r['conditions'].items():
                    timing=row['measurements']['model_call_seconds']
                    lines.append(f"|{condition}|{row['metrics']['soft']['p@5']:.4f}|{row['metrics']['hard']['p@5']:.4f}|{timing['mean']:.3f}|{timing['p95']:.3f}|{row['prompt_tokens']['mean']:.0f}|{len(row['failures'])}|")
                lines += ['',r['limitations'],'']
    p=Path('docs/exploration/teacher-compiled-lm.json')
    if p.exists():
        r=json.loads(p.read_text())
        lines += ['## Exact Compilation of the Full Teacher Corpus','',
                  f"{r['evidence_records']:,} records, {r['scored_profiles']:,} teachers with usable evidence; {r['queries']} original requirements, {r.get('repetitions',1)} repetitions.",
                  f"Mean query time: original {1000*r['mean_reference_seconds']:.3f}ms, compiled {1000*r['mean_compiled_seconds']:.3f}ms.",
                  r['warning'],'']
    lines += ['## New Synthetic Candidates and Condition-by-Condition Review','']
    for name,filename in (('Old lexical candidates','artifacts/synthesis-scope-v5/summary.json'),
                          ('Semantic candidates plus domain sources','artifacts/synthesis-semantic-v1/results/summary.json')):
        p=Path(filename)
        if p.exists():
            r=json.loads(p.read_text())
            lines.append(f"- {name}: {r['jobs']} constructions, {r['statuses'].get('silver_candidate',0)} candidates in initial review; not the final number approved in review.")
    p=Path('artifacts/scoped-atomic-audit-v1/summary.json')
    lines += [json.dumps(json.loads(p.read_text()),ensure_ascii=False) if p.exists() else 'Condition-by-condition review has no complete all-data summary yet; no inference is made about the final retained count.','']
    lines+=['## Limits of the Conclusions','','It has not yet been demonstrated that the new combined method generally improves both speed and accuracy over strong baselines. Classic methods, engineering compilation, public-dataset labels, and industry sources are reported separately; data candidates remain silver and are not combined into an overall gain.']
    write_english_report(output,'\n'.join(lines)+'\n');return output

if __name__=='__main__':print(generate())
