"""Normalize local records without indexing contact or demographic information."""
import json
from collections import Counter
from pathlib import Path
from .metadata.papers import publication_interval

FIELDS = ('\u7814\u7a76\u9886\u57df_CN','\u7814\u7a76\u9886\u57df_EN','\u4e2a\u4eba\u7b80\u4ecb_CN','\u4e2a\u4eba\u7b80\u4ecb_EN','\u8bba\u6587\u5217\u8868','\u9879\u76ee\u5217\u8868','\u4e13\u5229\u5217\u8868')

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
            experts[eid] = dict(id=eid, name=row.get('\u6559\u5e08\u59d3\u540d_CN') or row.get('\u6559\u5e08\u59d3\u540d_EN'),
                institution=row.get('\u673a\u6784\u540d_CN',''), college=row.get('\u5b66\u9662\u540d_CN',''),
                url=row.get('\u6559\u5e08\u8be6\u60c5\u9875\u94fe\u63a5\u6216\u8bbf\u95ee\u65b9\u5f0f_CN',''))
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
            title=row.get('title'), url=row.get('gs_link'), source=f'{root}/papers.jsonl:{line}',
            publication=publication_interval(row.get('gs_publication_date')),
            contribution_status='unverified', authorship_association='provided_teacher_id_not_independently_verified'))
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
