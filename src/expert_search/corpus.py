"""Normalize local records without indexing contact or demographic information."""
import json
from collections import Counter
from pathlib import Path

FIELDS = ('研究领域_CN','研究领域_EN','个人简介_CN','个人简介_EN','论文列表','项目列表','专利列表')

def records(path):
    with Path(path).open(encoding='utf-8') as f:
        for line, text in enumerate(f, 1):
            if text.strip():
                try:
                    yield line, json.loads(text)
                except ValueError as e:
                    raise ValueError(f'{path}:{line}: {e}') from e

def build(data, output):
    root = Path(data) / 'teachers/shenzhen'
    experts, evidence, duplicates = {}, [], []
    for path in sorted(root.glob('*.jsonl')):
        if path.name == 'papers.jsonl':
            continue
        for line, row in records(path):
            eid = row['teacher_id']
            if eid in experts:
                duplicates.append(eid)
                continue
            experts[eid] = dict(id=eid, name=row.get('教师姓名_CN') or row.get('教师姓名_EN'),
                institution=row.get('机构名_CN',''), college=row.get('学院名_CN',''),
                url=row.get('教师详情页链接或访问方式_CN',''))
            for key in FIELDS:
                value = row.get(key)
                if value:
                    evidence.append(dict(id=f'{eid}:{key}', expert_id=eid, text=str(value),
                        kind='profile', field=key, source=f'{path.as_posix()}:{line}'))
    orphan = 0
    for line, row in records(root/'papers.jsonl'):
        eid = row['teacher_id']
        if eid not in experts:
            orphan += 1
            continue
        evidence.append(dict(id=f'paper:{line}', expert_id=eid, kind='paper',
            text='\n'.join(str(row.get(k) or '') for k in ('title','gs_abstract')),
            title=row.get('title'), url=row.get('gs_link'), source=f'{root}/papers.jsonl:{line}'))
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    for name, rows in [('experts',list(experts.values())),('evidence',evidence)]:
        with (out/f'{name}.jsonl').open('w') as f:
            for row in rows:
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
    report = dict(experts=len(experts), evidence=len(evidence), duplicate_ids=duplicates,
        orphan_papers=orphan, kinds=dict(Counter(x['kind'] for x in evidence)),
        institutions=len(set(x['institution'] for x in experts.values())))
    (out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report
