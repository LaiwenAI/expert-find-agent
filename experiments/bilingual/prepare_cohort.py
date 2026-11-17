"""New company-grouped raw demand cohort for large bilingual construction.

English fields start pending; this preparer does not pretend translations/labels exist.
"""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import random
import unicodedata
from openpyxl import load_workbook
from experiments.prepare_fresh_method_data import normalized_hash as conservative_prior_hash

ROOT=Path('artifacts/large-bilingual-data-v1')


def exact_key(text):
    # Preserve comparator, decimal punctuation and Latin unit case (mM != mm).
    return ' '.join(unicodedata.normalize('NFKC',text).split())


def company_key(text):return exact_key(text).casefold()


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    parent=Path('artifacts/fresh-method-data-v1');selection=json.loads((parent/'selection.json').read_text());registry=json.loads((parent/'prior-use.json').read_text());old=list(map(json.loads,(parent/'queries.jsonl').read_text().splitlines()))
    workbook=Path('data/\u771f\u5b9e\u4f01\u4e1a\u9700\u6c42/\u5168\u56fd\u5927\u5b66\u751f\u521b\u4e1a\u670d\u52a1\u7f51_\u4ea7\u4e1a\u547d\u9898.xlsx')
    if sha(workbook)!=selection['workbook_sha256']:raise ValueError('Changed original workbook')
    old_ids=set(registry['demand_ids'])|{r['id'] for r in old};old_aliases=set(registry['query_hashes'])
    for r in old:
        old_aliases.add(conservative_prior_hash(r['demand']));old_aliases.add(conservative_prior_hash(r['demand'].split('\n',1)[1]))
    book=load_workbook(workbook,read_only=True,data_only=True);lines=iter(book.active.values);header=list(next(lines));rows=[]
    for line,values in enumerate(lines,2):
        r=dict(zip(header,values));title=r.get('\u547d\u9898\u540d\u79f0');body=r.get('\u547d\u9898\u5185\u5bb9\u53ca\u7b54\u9898\u8981\u6c42')
        if not title or not body:continue
        text=str(title)+'\n'+str(body);rows.append(dict(id='demand-'+str(line),zh=text,title=str(title),company=str(r.get('\u547d\u9898\u4f01\u4e1a') or ''),category=str(r.get('\u547d\u9898\u7c7b\u522b') or ''),track=str(r.get('\u547d\u9898\u7ec4\u522b') or ''),url=r.get('\u8be6\u60c5\u94fe\u63a5'),source_row=line))
    book.close();prior_companies={company_key(r['company']) for r in rows if r['id'] in old_ids and company_key(r['company'])};excluded=Counter();selected=[];seen={};aliases=[]
    for r in rows:
        company=company_key(r['company']);body=r['zh'].split('\n',1)[1]
        if r['id'] in old_ids:excluded['prior_demand_id']+=1;continue
        if company and company in prior_companies:excluded['prior_company_exact_key']+=1;continue
        if {conservative_prior_hash(r['zh']),conservative_prior_hash(body)}&old_aliases:excluded['conservative_prior_text_alias']+=1;continue
        key=exact_key(r['zh'])
        if key in seen:aliases.append(dict(alias_id=r['id'],canonical_id=seen[key],source_row=r['source_row']));excluded['exact_full_text_duplicate']+=1;continue
        seen[key]=r['id'];selected.append(r)
    parent_id={r['id']:r['id'] for r in selected}
    def find(x):
        while parent_id[x]!=x:parent_id[x]=parent_id[parent_id[x]];x=parent_id[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent_id[max(a,b)]=min(a,b)
    keys={}
    for r in selected:
        candidates=[('body',exact_key(r['zh'].split('\n',1)[1]))]
        if company_key(r['company']):candidates.append(('company',company_key(r['company'])))
        if isinstance(r['url'],str) and r['url'].strip():candidates.append(('url',r['url'].strip()))
        for key in candidates:
            if key in keys:union(r['id'],keys[key])
            else:keys[key]=r['id']
    groups=defaultdict(list)
    for r in selected:groups[find(r['id'])].append(r)
    components=sorted(groups.values(),key=lambda g:min(r['id'] for r in g));random.Random('large-bilingual-v1:20261117').shuffle(components)
    splits={};counts=Counter();n=len(selected)
    for group in components:
        split='construction' if counts['construction']<round(.8*n) else 'validation' if counts['validation']<round(.1*n) else 'sealed'
        gid=hashlib.sha256('\n'.join(sorted(r['id'] for r in group)).encode()).hexdigest()
        for r in group:splits[r['id']]=(split,gid)
        counts[split]+=len(group)
    outputs=[]
    for r in selected:
        split,gid=splits[r['id']];outputs.append(dict(**r,split=split,group_id=gid,en=None,translation_status='pending_not_bilingual_yet',expert_fit_grade=None,source=dict(file=str(workbook),file_sha256=selection['workbook_sha256'],row=r['source_row'],text_sha256=hashlib.sha256(r['zh'].encode()).hexdigest())))
    body=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in outputs);aliasbody=''.join(json.dumps(r)+'\n' for r in aliases)
    protocol=dict(inputs_sha256={str(p):sha(p) for p in [workbook,parent/'selection.json',parent/'prior-use.json',parent/'queries.jsonl']},seed='large-bilingual-v1:20261117',selection='Original full demands; exclude prior IDs, exact prior-company keys and conservative historical text aliases. New exact normalization preserves comparators/decimal punctuation/unit case. Group by exact company/body/URL before translation or labels;80/10/10 by grouped demand counts.',code_sha256=sha(Path(__file__)),outputs_sha256=dict(queries=hashlib.sha256(body.encode()).hexdigest(),duplicate_aliases=hashlib.sha256(aliasbody.encode()).hexdigest()),scope='Large raw-derived construction plan. English translations and expert labels pending, not a completed bilingual/gold dataset. Company families/aliases and semantic duplicate independence unverified. Original100cohort including sealed20 excluded. Real full texts remain local.')
    ROOT.mkdir(exist_ok=True);marker=ROOT/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed large bilingual preparation')
    marker.write_text(json.dumps(protocol,indent=2));(ROOT/'queries-zh.jsonl').write_text(body);(ROOT/'duplicate-aliases.jsonl').write_text(aliasbody)
    report=dict(original_nonempty_rows=len(rows),new_original_demands=len(outputs),prior_company_keys=len(prior_companies),excluded=dict(excluded),groups=len(components),largest_group=max(map(len,components),default=0),splits=dict(counts),categories=dict(Counter(r['category'] for r in outputs)),english_translations_completed=0,expert_labels_completed=0,prior100_query_overlap=len({r['id'] for r in old}&{r['id'] for r in outputs}),protocol_sha256=sha(marker),scope=protocol['scope'])
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False));print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__':run()
