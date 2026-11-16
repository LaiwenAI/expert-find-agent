"""Separate comparable timing boundaries instead of mixing models and stages."""
import json
from pathlib import Path


def generate(output='docs/paper/quality-time.md'):
    lines=['# 质量—时间证据表','','不同任务、模型和计时范围分表；不将局部计算速度当作 agent 端到端速度。','',
        '## 同编码器、本地检索阶段','','|路径|SciFact nDCG@10|均值 ms|p95 ms|','|---|---:|---:|---:|']
    p=Path('docs/exploration/progressive-latency.json')
    if p.exists():
        r=json.loads(p.read_text())
        for name in ['full','progressive']:
            t=r['timings'][name];m=r['quality'][name]
            lines.append(f"|{name}|{m['ndcg@10']:.4f}|{t['mean_seconds']*1000:.3f}|{t['p95_seconds']*1000:.3f}|")
    lines+=['','上述均预计算查询向量，不能表示用户在线延迟。','',
        '## 查询编码探测','','|路径|样本|p50 ms|p95 ms|质量边界|','|---|---:|---:|---:|---|']
    p=Path('docs/exploration/local-query-latency.json')
    if p.exists():
        r=json.loads(p.read_text());lines.append(f"|本地冻结 BGE|{r['samples']}|{r['p50_seconds']*1000:.2f}|{r['p95_seconds']*1000:.2f}|完整精度单独测量|")
    lines+=['','API 编码五条样本约 0.52–1.40 秒；不能用这五条与本地二十条推断严格 p95 加速。模型不同，精度也不能拼接。','',
        '## 本地完整流程','']
    p=Path('artifacts/public/scifact-local-bge/summary.json')
    if p.exists():
        r=json.loads(p.read_text());lines.append(f"300 查询：nDCG@10={r['metrics']['ndcg@10']:.4f}，编码＋检索 p50={r['p50_seconds']*1000:.2f}ms、p95={r['p95_seconds']*1000:.2f}ms。")
    else:lines.append('语料编码仍在执行，尚无完整流程的质量—时间结果。')
    lines+=['','## 推理路径','','CMU 与 SciFact 是不同任务，不作跨任务 Pareto 支配比较。在线核验包含模型生成，必须与原始画像基线同时报告精度、请求数、离线成本及模型延迟。']
    Path(output).write_text('\n'.join(lines)+'\n');return output

if __name__=='__main__':print(generate())
