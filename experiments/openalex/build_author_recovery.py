"""Build a public temporal known-author-recovery proxy from frozen OpenAlex metadata."""
import hashlib
import argparse
import json
import re
from collections import Counter,defaultdict
from pathlib import Path


NON_ENGLISH_MARKS=re.compile(r'[\u2e80-\u2fdf\u3000-\u303f\u3100-\u312f\u31a0-\u31bf\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef\U00020000-\U0003134f]')
ID=re.compile(r'https://openalex\.org/([WA]\d+)$')


def key(title):return re.sub(r'[^a-z0-9]+',' ',title.casefold()).strip()


def title_grams(title):
    text=key(title).replace(' ','')
    return {text[i:i+4] for i in range(len(text)-3)} if len(text)>=4 else set()


def similarity(a,b):return len(a&b)/len(a|b) if a|b else 0.


def record(row,period):
    title=row.get('title');year=row.get('publication_year');wid=ID.fullmatch(row.get('id') or '')
    if not isinstance(title,str) or not title.strip() or len(title)>500 or NON_ENGLISH_MARKS.search(title) or row.get('language')!='en':return None
    if not wid or not wid.group(1).startswith('W') or type(year)!=int:return None
    if not (2010<=year<=2019 if period=='history' else 2020<=year<=2023):return None
    authors=[]
    for a in row.get('authorships') or []:
        found=ID.fullmatch((a.get('author') or {}).get('id') or '')
        if found and found.group(1).startswith('A'):authors.append(found.group(1))
    authors=sorted(set(authors))
    if not authors or not key(title):return None
    return {'id':wid.group(1),'title':title.strip(),'year':year,'authors':authors,'period':period,'language':'en'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('benchmarks/openalex-author-recovery-v1'));args=parser.parse_args()
    source=Path('artifacts/public/openalex-api-snapshot-v1');snapshot=json.loads((source/'manifest.json').read_text());rows={};duplicates=0
    for item in snapshot['records']:
        path=source/item['file'];raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Changed API snapshot')
        period='history' if item['file'].endswith('-history.json') else 'target'
        for row in json.loads(raw)['results']:
            value=record(row,period)
            if value is None:continue
            if value['id'] in rows:
                if rows[value['id']]!=value:raise ValueError('Conflicting duplicate work metadata')
                duplicates+=1
            rows[value['id']]=value
    target_all=sorted((r for r in rows.values() if r['period']=='target'),key=lambda r:r['id']);target_titles={key(r['title']) for r in target_all}
    unique_targets={}
    for row in target_all:unique_targets.setdefault(key(row['title']),row)
    targets=list(unique_targets.values())
    target_grams=[title_grams(r['title']) for r in target_all]
    eligible_history=[r for r in rows.values() if r['period']=='history' and key(r['title']) not in target_titles]
    history_all=sorted((r for r in eligible_history for grams in [title_grams(r['title'])]
                        if all(similarity(grams,g)<.8 for g in target_grams)),key=lambda r:(r['year'],r['id']))
    unique_history={}
    for row in history_all:unique_history.setdefault((key(row['title']),tuple(row['authors'])),row)
    history=list(unique_history.values())
    if {r['id'] for r in history}&{r['id'] for r in targets}:raise ValueError('Temporal overlap')
    authors={aid for r in history for aid in r['authors']};queries=[];qrels=[];unavailable=0
    for r in sorted(targets,key=lambda x:x['id']):
        positives=sorted(set(r['authors'])&authors);unavailable+=len(set(r['authors'])-authors)
        if not positives:continue
        queries.append({'id':r['id'],'text':r['title'],'publication_year':r['year']})
        qrels.append({'query_id':r['id'],'known_author_ids':positives})
    evidence=[]
    for work in sorted(history,key=lambda x:x['id']):
        for aid in work['authors']:
            evidence.append({'id':work['id']+':'+aid,'expert_id':aid,'text':work['title'],
                             'source_work_id':work['id'],'publication_year':work['year']})
    experts=[{'id':aid} for aid in sorted(authors)]
    if not queries or not evidence:raise ValueError('No usable benchmark records')
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    for name,values in [('experts',experts),('evidence',evidence),('queries',queries),('qrels',qrels),('source-records',sorted(history+targets,key=lambda x:x['id']))]:
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=True,sort_keys=True)+'\n' for r in values))
    outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.jsonl')}
    manifest={'benchmark':'openalex-author-recovery-v1','queries':len(queries),'experts':len(experts),'evidence':len(evidence),
              'history_works':len(history),'target_works':len(targets),'discarded_duplicate_api_rows':duplicates,
              'discarded_duplicate_target_titles':len(target_all)-len(targets),'discarded_duplicate_history_records':len(history_all)-len(history),
              'discarded_history_high_title_overlap':len(eligible_history)-len(history_all),
              'target_authors_without_historical_evidence':unavailable,'periods':{'history':'2010-2019','target':'2020-2023'},
              'input_snapshot_manifest_sha256':hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
              'api_sources':snapshot['records'],'fetched_utc':snapshot['fetched_utc'],'outputs_sha256':outputs,
              'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope':'English-labeled publication-title known-author recovery, using OpenAlex author associations; not enterprise relevance, ability, individual contribution, or exhaustive labels. Other experts are unjudged. Exact title matches and >=0.8 character 4-gram Jaccard matches against target titles are excluded; semantic leakage remains possible. Threshold was chosen after observing the preliminary snapshot. Snapshot API results can change; filtered source-records and hashes are distributed, raw responses are not.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('api_sources','outputs_sha256')},indent=2))


if __name__=='__main__':main()
