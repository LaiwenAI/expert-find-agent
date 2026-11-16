"""Generate a comparison restricted to common queries; include failures as zero."""
import json
import random
from pathlib import Path


def generate(baseline='artifacts/benchmark', agent='artifacts/agent-benchmark', output='docs/benchmark-results.md'):
    base=Path(baseline); model=Path(agent)
    manifest=json.loads((base/'manifest.json').read_text())
    rows=[json.loads(x) for x in (base/'runs.jsonl').read_text().splitlines()]
    agent_rows=[json.loads(p.read_text()) for p in model.glob('paper-*.json')]
    common={x['query_id'] for x in agent_rows}
    lines=['# Benchmark 结果','',
        '任务：留出论文标题→已知关联专家恢复。代理标签来自论文归属，不是人工需求相关性。',
        f"语料 SHA256：`{manifest['corpus_sha256']}`；随机种子 {manifest['seed']}。",'',
        '## 全部基线查询','', '| 方法 | 查询数 | nDCG@10 | MRR@10 | Recall@10 | 平均秒数 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    def table(data):
        for method in sorted({x['method'] for x in data}):
            rs=[x for x in data if x['method']==method]; n=len(rs)
            means=[sum(x['metrics'][k] for x in rs)/n for k in ['ndcg@10','mrr@10','recall@10']]
            lines.append(f'| {method} | {n} | '+ ' | '.join(f'{v:.4f}' for v in means)+f" | {sum(x['seconds'] for x in rs)/n:.3f} |")
    table(rows)
    if common:
        lines.extend(['','## 同一模型评测查询子集','',
            '| 方法 | 查询数 | nDCG@10 | MRR@10 | Recall@10 | 平均秒数 |',
            '| --- | ---: | ---: | ---: | ---: | ---: |'])
        normalized=[dict(x,method='hybrid-evidence-tree') for x in agent_rows]
        table([x for x in rows if x['query_id'] in common]+normalized)
        lines.extend(['',f"模型实验失败数：{sum(x['status']!='ok' for x in agent_rows)}。失败按零分计入，不能只汇总成功请求。"])
        if len(common)>=2:
            lines.extend(['','配对 bootstrap（2,000 次，seed=42）：树 agent 减 BM25-vote 的 nDCG@10 差值。'])
            a={r['query_id']:r['metrics']['ndcg@10'] for r in agent_rows}
            b={r['query_id']:r['metrics']['ndcg@10'] for r in rows if r['method']=='bm25-vote'}
            differences=[a[q]-b[q] for q in sorted(common)]
            rng=random.Random(42)
            estimates=sorted(sum(rng.choices(differences,k=len(differences)))/len(differences) for _ in range(2000))
            lines.append(f'平均差 {sum(differences)/len(differences):.4f}，95% bootstrap 区间 [{estimates[49]:.4f}, {estimates[1949]:.4f}]。小样本结果不应外推。')
        successful=[x for x in agent_rows if x['status']=='ok']
        if successful:
            prompt=sum(x['response']['usage']['prompt_tokens'] for x in successful)
            completion=sum(x['response']['usage']['completion_tokens'] for x in successful)
            recall=sum(len(set(x['response']['candidate_ids'])&{e for e,g in x['qrels'].items() if g>0})/sum(g>0 for g in x['qrels'].values()) for x in successful)/len(successful)
            lines.extend(['',f'成功请求总 prompt tokens：{prompt}；completion tokens：{completion}。未配置价格表，不估算金额。',f'成功请求的候选集已知作者召回率：{recall:.4f}；这约束后续排序可达上限。'])
    lines.extend(['','## 解释限制','',
        '作者归属不包含所有相关专家；Recall 是已知标签恢复率。标题对应证据及所有包含该完整标题的字段均被排除；相关论文和改写内容仍可提供信息。集合统计仍包含被排除文档。',
        '基线在同一进程顺序运行，耗时包含缓存效应；模型耗时还受服务负载影响。没有人工 gold 和统计显著性证据，不能据此声称真实企业需求上的优越性。',
        '树 agent 的首阶段候选受 BM25 限制，后续模型查证可能改善解释，也可能损失排名；需结合逐查询结果检查。',
        '原始数据：artifacts/benchmark/runs.jsonl；模型阶段输出和来源：artifacts/agent-benchmark/paper-*.json。'])
    Path(output).write_text('\n'.join(lines)+'\n')
    return output
