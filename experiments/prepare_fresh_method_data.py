"""Freeze a genuinely new demand cohort and an explicit prior-use exclusion registry."""
import hashlib
import json
from pathlib import Path
import random
import re
from openpyxl import load_workbook

ROOT=Path('artifacts/fresh-method-data-v1')


def normalized_hash(text):
    return hashlib.sha256(re.sub(r'[^\w]+','',text.casefold()).encode()).hexdigest()


def run():
    ROOT.mkdir(exist_ok=True);registry_path=ROOT/'prior-use.json'
    if registry_path.exists():
        registry=json.loads(registry_path.read_text())
        for path,expected in registry['parents'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise ValueError('Changed prior-use parent')
    else:
        paths=sorted(p for p in Path('artifacts').rglob('*.jsonl') if p.name in {'jobs.jsonl','queries.jsonl','examples.jsonl','cases.jsonl'} and not str(p).startswith(('artifacts/public/',str(ROOT)+'/')))
        demand_ids=set();expert_ids=set();query_hashes=set();source_hashes=set();normalized_sources=set();paper_titles=set()
        def visit(value):
            if isinstance(value,dict):
                if isinstance(value.get('text'),str) and isinstance(value.get('expert_id'),str):
                    text=value['text'][:1200];source_hashes.add(hashlib.sha256(text.encode()).hexdigest());normalized_sources.add(normalized_hash(text))
                    if value.get('kind')=='paper' or str(value.get('id','')).startswith('paper:'):
                        for title in [value.get('title',''),text.splitlines()[0] if text else '']:
                            if isinstance(title,str) and len(re.sub(r'[^\w]+','',title))>=12:paper_titles.add(normalized_hash(title))
                for key,child in value.items():
                    if key in {'demand','original_demand','query'} and isinstance(child,str):
                        query_hashes.add(normalized_hash(child))
                        if '\n' in child:query_hashes.add(normalized_hash(child.split('\n',1)[1]))
                    visit(key);visit(child)
            elif isinstance(value,list):
                for child in value:visit(child)
            elif isinstance(value,str):
                demand_ids.update(re.findall(r'demand-\d+',value))
                if re.fullmatch(r'CN_[A-Za-z0-9_-]+',value):expert_ids.add(value)
        for path in paths:
            for line in path.read_text().splitlines():visit(json.loads(line))
        registry=dict(parents={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},demand_ids=sorted(demand_ids),expert_ids=sorted(expert_ids),query_hashes=sorted(query_hashes),source_prefix_hashes=sorted(source_hashes),normalized_source_hashes=sorted(normalized_sources),paper_title_hashes=sorted(paper_titles),
                      scope='Explicit prior selected/evaluated local nonpublic job/query/case/example JSONL files; not the whole indexed corpus. Recursive IDs plus normalized full demand/body, source prefixes and exact paper-title aliases. Does not establish no semantic or team-family leakage.')
        registry_path.write_text(json.dumps(registry,indent=2))
    workbook=next(Path('data/真实企业需求').glob('*.xlsx'))
    book=load_workbook(workbook,read_only=True,data_only=True);sheet=book.active;lines=iter(sheet.values);header=list(next(lines));pool=[]
    used_ids=set(registry['demand_ids']);used_hashes=set(registry['query_hashes']);seen=set()
    for line,values in enumerate(lines,2):
        record=dict(zip(header,values));title=record.get('命题名称');body=record.get('命题内容及答题要求')
        if not title or not body:continue
        title,body=str(title),str(body);text=title+'\n'+body;key='demand-'+str(line)
        hashes={normalized_hash(text),normalized_hash(body)}
        if key in used_ids or hashes&used_hashes or hashes&seen:continue
        seen.update(hashes);pool.append(dict(id=key,demand=text,title=title,url=record.get('详情链接'),source_row=line))
    book.close();random.Random(20261117).shuffle(pool);queries=pool[:100]
    if len(queries)!=100:raise ValueError('Insufficient new demands under frozen exclusions')
    body=''.join(json.dumps(q,ensure_ascii=False)+'\n' for q in queries);path=ROOT/'queries.jsonl'
    if path.exists() and path.read_text()!=body:raise ValueError('Changed fresh demand cohort')
    path.write_text(body)
    protocol=dict(seed=20261117,queries=100,available_new_unique_demands=len(pool),prior_demands=len(used_ids),prior_experts=len(registry['expert_ids']),prior_source_prefixes=len(registry['source_prefix_hashes']),
                  workbook_sha256=hashlib.sha256(workbook.read_bytes()).hexdigest(),registry_sha256=hashlib.sha256(registry_path.read_bytes()).hexdigest(),queries_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  selection='First100 after fixed-seed shuffle of unseen original full demands; no outcome or teacher-topic filtering. Retrieval excludes all registered old expert IDs and exact source/title aliases. Split by connected expert/source groups before any generation labels.',
                  scope='New source cohort, not yet labelled expert-fit gold. Original full demands retained. Near-semantic duplicates and institution/team families still require audit. Real texts remain local, not in software release.')
    marker=ROOT/'selection.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed new data selection')
    marker.write_text(json.dumps(protocol,indent=2));print(json.dumps({k:v for k,v in protocol.items() if k not in {'selection','scope'}},indent=2))


if __name__=='__main__':run()
