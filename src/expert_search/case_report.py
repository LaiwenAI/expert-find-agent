import json
from pathlib import Path


def generate(output='docs/industrial-cases.md'):
    lines=['# 真实企业需求案例','',
        '以下是系统运行实例，不是人工相关性评测。推荐理由及缺失项均由模型生成。候选人可能仅具备邻近能力，不能据此认定已满足企业技术指标。','']
    for p in sorted(Path('artifacts/demand-cases').glob('*.json')):
        r=json.loads(p.read_text()); q=r['query']
        lines.extend([f"## {q['title']}",'',f"需求来源：{q['source']}",f"详情：{q.get('url','')}",'',f"运行状态：{r['status']}；总耗时 {r['seconds']:.2f} 秒。",''])
        for method,result in r['baselines'].items():
            lines.append(f"- {method}："+'、'.join(x['name'] or x['id'] for x in result))
        for x in r.get('agent',{}).get('results',[]):
            lines.extend(['',f"### {x['expert']['name']}",'',x['reason'],'',
                '未获证据支持的要求：'+'；'.join(x['missing']) if x['missing'] else '模型未列出缺失项，仍需人工核验。',''])
            for e in x['evidence']:
                lines.append(f"- `{e['id']}`，来源 `{e['source']}`：{e['text'][:350].replace(chr(10),' ')}")
        lines.extend(['',f'完整结果及模型 trace：`{p}`。',''])
    Path(output).write_text('\n'.join(lines)+'\n')
    return output
