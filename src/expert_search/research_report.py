"""Artifact-derived tables for the research draft; missing results stay missing."""
import json
from pathlib import Path


def generate(output='docs/paper/results-generated.md'):
    lines=['# 自动生成的实验结果','', '本文件只读取已有实验产物；没有文件的实验不报告结果。','']
    p=Path('artifacts/public/scifact-results/summary.json')
    if p.exists():
        r=json.loads(p.read_text());lines+=['## SciFact 词法基线','','|方法|查询数|nDCG@10|MRR@10|Recall@10|p50 秒|p95 秒|','|---|---:|---:|---:|---:|---:|---:|']
        for name,x in r['results'].items():
            m=x['metrics'];lines.append(f"|{name}|{x['queries']}|{m['ndcg@10']:.4f}|{m['mrr@10']:.4f}|{m['recall@10']:.4f}|{x['p50_seconds']:.4f}|{x['p95_seconds']:.4f}|")
        lines+=['','只包含本地词法查询，不含模型 API。','']
    p=Path('artifacts/public/scifact-hybrid-results/summary.json')
    if p.exists():
        r=json.loads(p.read_text());lines+=['## SciFact 冻结语义与混合检索','',f"模型：{r['model']}。{r['latency_warning']}",'']
        for method,m in r['results'].items():lines.append(f"- {method}：nDCG@10={m['ndcg@10']:.4f}，Recall@10={m['recall@10']:.4f}。")
    else:lines+=['## SciFact 语义实验','','完整结果尚未生成，不报告部分语料分数。','']
    p=Path('artifacts/public/cmu-results/lexical-summary.json')
    if p.exists():
        r=json.loads(p.read_text());lines+=['## CMU Gold 单档案版本','',f"词法基线加权配对损失：{r['weighted_pairwise_loss']:.6f}，比较对数：{r['comparable_pairs']}。",'']
    p=Path('artifacts/public/cmu-dense-results/summary.json')
    if p.exists():
        r=json.loads(p.read_text())
        for method,m in r['results'].items():lines.append(f"- {method}：损失 {m['weighted_pairwise_loss']:.6f}。")
        lines+=['',r['warning'],'']
    else:lines+=['冻结语义画像实验尚未完成。','']
    p=Path('artifacts/synthetic-evidence-first-selected/selection.json')
    if p.exists():
        r=json.loads(p.read_text());lines+=['## 合成数据筛选','',f"证据优先批次 {r['total']} 次尝试，保留 {r['accepted']} 条。保留标签仍为 silver，完整查询及反事实语义未获充分验证。",'']
    for dataset in ('sigir','nips'):
        p=Path(f'docs/exploration/compiled-lm-{dataset}.json')
        if p.exists():
            r=json.loads(p.read_text())
            lines += [f'## {dataset.upper()} 经典候选语言模型与预编译','',
                      '|方法|Soft P@5|Hard P@5|本地均值 ms|','|---|---:|---:|---:|']
            for method,m in r['metrics']['soft'].items():
                lines.append(f"|{method}|{m['p@5']:.4f}|{r['metrics']['hard'][method]['p@5']:.4f}|{1000*r['timings'][method]['mean']:.3f}|")
            lines += ['',f"{r['queries']} 个查询，每条三次交替随机顺序；编译前后分数完全一致。质量属于经典模型，编译仅改变执行。",r['warning'],'']
        p=Path(f'docs/exploration/cof-{dataset}-packing-shared-analysis.json')
        if p.exists():
            r=json.loads(p.read_text())
            if r.get('complete'):
                lines += [f'## {dataset.upper()} 同候选共享证据实验','',
                          '|证据|Soft P@5|Hard P@5|调用均值 s|P95 s|平均输入 token|失败|','|---|---:|---:|---:|---:|---:|---:|']
                for condition,row in r['conditions'].items():
                    timing=row['measurements']['model_call_seconds']
                    lines.append(f"|{condition}|{row['metrics']['soft']['p@5']:.4f}|{row['metrics']['hard']['p@5']:.4f}|{timing['mean']:.3f}|{timing['p95']:.3f}|{row['prompt_tokens']['mean']:.0f}|{len(row['failures'])}|")
                lines += ['',r['limitations'],'']
    p=Path('docs/exploration/teacher-compiled-lm.json')
    if p.exists():
        r=json.loads(p.read_text())
        lines += ['## 完整教师语料的精确编译','',
                  f"{r['evidence_records']:,} 条记录、{r['scored_profiles']:,} 位有可用证据教师；{r['queries']} 条原始需求，{r.get('repetitions',1)} 次重复。",
                  f"查询均值：原始 {1000*r['mean_reference_seconds']:.3f}ms，编译 {1000*r['mean_compiled_seconds']:.3f}ms。",
                  r['warning'],'']
    lines += ['## 新合成候选与逐条件复审','']
    for name,filename in (('旧词法候选','artifacts/synthesis-scope-v5/summary.json'),
                          ('语义候选加领域来源','artifacts/synthesis-semantic-v1/results/summary.json')):
        p=Path(filename)
        if p.exists():
            r=json.loads(p.read_text())
            lines.append(f"- {name}：{r['jobs']} 次构造，初审候选 {r['statuses'].get('silver_candidate',0)}；不是最终复审通过数。")
    p=Path('artifacts/scoped-atomic-audit-v1/summary.json')
    lines += [json.dumps(json.loads(p.read_text()),ensure_ascii=False) if p.exists() else '逐条件复审尚无全量完成汇总，不推断最终保留数。','']
    lines+=['## 结论边界','','目前未证明新组合方法在强基线之上普遍同时提速提准。经典方法、工程编译、公开数据标签与产业来源支持分别报告；数据候选保持 silver，不混合为总体收益。']
    Path(output).write_text('\n'.join(lines)+'\n');return output

if __name__=='__main__':print(generate())
