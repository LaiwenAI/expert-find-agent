"""Construct and run a private temporal author-recovery proxy from local data."""
import hashlib
import itertools
import sqlite3
import json
import math
import random
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from expert_search.corpus import records
from expert_search.metadata.papers import availability
from expert_search.retrieval import Search

SOURCE=Path('artifacts/corpus-temporal-source-v1')
HISTORY=Path('artifacts/corpus-asof-2020-v1')
DATABASE=Path('artifacts/temporal-search-2020-v1.sqlite')
OUTPUT=Path('artifacts/local-temporal-author-proxy-v1')


def normalize(text):
    return ''.join(re.findall(r'[a-z0-9\u4e00-\u9fff]',text.lower()))


def shingles(text):
    value=normalize(text)
    return {value[i:i+4] for i in range(len(value)-3)}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest=json.loads((HISTORY/'temporal-manifest.json').read_text())
    if manifest['as_of']!='2020-12-31' or sha(SOURCE/'evidence.jsonl')!=manifest['source_evidence_sha256'] or sha(HISTORY/'evidence.jsonl')!=manifest['output_evidence_sha256']:
        raise ValueError('Changed historical corpus')
    historical=[r for _,r in records(HISTORY/'evidence.jsonl')]
    with sqlite3.connect('file:' + str(DATABASE) + '?mode=ro', uri=True) as db:
        for source, stored in itertools.zip_longest(enumerate(historical), db.execute('SELECT id,body FROM docs ORDER BY id')):
            if source is None or stored is None or source[0] != stored[0] or source[1] != json.loads(stored[1]):
                raise ValueError('Index documents differ from historical corpus')
    authors={r['expert_id'] for r in historical}; histories={}
    for row in historical:
        title=row.get('title') or ''
        if title.strip():histories.setdefault(normalize(title),shingles(title))
    inverted=defaultdict(set)
    for title,grams in histories.items():
        for gram in grams:inverted[gram].add(title)
    targets={}
    for _,row in records(SOURCE/'evidence.jsonl'):
        if row.get('kind')!='paper' or not isinstance(row.get('publication'),dict):continue
        if availability(row['publication'],'2020-12-31')!='after_cutoff' or availability(row['publication'],'2023-12-31')!='before_or_on_cutoff':continue
        title=row.get('title') or ''
        if not isinstance(title,str) or len(normalize(title))<12:continue
        key=normalize(title)
        item=targets.setdefault(key,dict(title=title,authors=set()))
        item['authors'].add(row['expert_id'])
    eligible=[];unavailable=0;leakage=0
    ordered=sorted(targets.items());random.Random(20261117).shuffle(ordered)
    for title,row in ordered:
        positives=sorted(row['authors'] & authors)
        if not positives:unavailable+=1;continue
        grams=shingles(row['title']);candidates=set()
        for gram in grams:candidates.update(inverted.get(gram,()))
        if any(len(grams & histories[h])/len(grams | histories[h])>=.8 for h in candidates if grams | histories[h]):
            leakage+=1;continue
        eligible.append((title,row,positives))
        if len(eligible)==200: break
    selected=eligible
    if len(selected)!=200:raise ValueError('Fewer than 200 eligible titles')
    OUTPUT.mkdir(parents=True,exist_ok=False)
    engine=Search(HISTORY,DATABASE);runs=[]
    for i,(_,row,truth) in enumerate(selected):
        qid='q-%04d'%i;truth=set(truth);methods={}
        for method in (('bm25-max','bm25-vote') if i%2==0 else ('bm25-vote','bm25-max')):
            start=time.perf_counter();hits=engine.search(row['title'],10,method);seconds=time.perf_counter()-start
            rank=[r['id'] for r in hits];positions=[n for n,eid in enumerate(rank,1) if eid in truth]
            methods[method]=dict(seconds=seconds,hit_at_10=int(bool(positions)),mrr_at_10=1/min(positions) if positions else 0,
                                 known_author_recall_at_10=len(set(rank)&truth)/len(truth),ranking=rank)
        runs.append(dict(id=qid,title=row['title'],known_author_ids=sorted(truth),methods=methods))
    engine.db.close()
    data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in runs).encode();(OUTPUT/'results.jsonl').write_bytes(data)
    summary=dict(as_of='2020-12-31',target_period='2021-2023',target_titles=len(targets),assessed_titles=len(eligible)+unavailable+leakage,assessed_unavailable_titles=unavailable,
                 assessed_high_overlap_excluded=leakage,sampled_queries=len(selected),history_authors=len(authors),
                 methods={m:{k:statistics.mean(r['methods'][m][k] for r in runs) for k in ('seconds','hit_at_10','mrr_at_10','known_author_recall_at_10')} for m in ('bm25-max','bm25-vote')},
                 source_sha256=sha(SOURCE/'evidence.jsonl'),historical_sha256=sha(HISTORY/'evidence.jsonl'),database_sha256=sha(DATABASE),results_sha256=hashlib.sha256(data).hexdigest(),
                 scope='Local private proxy from supplied teacher-paper associations, not independently verified identities, individual contributions, or enterprise suitability. Known positives are target-associated authors with selected history; others unjudged. Exact normalized title grouping and char-four-gram Jaccard >=0.8 historical overlap filter do not remove all semantic leakage. Title/date metadata may be wrong. Fixed-seed random ordering over target titles with rejection until 200 eligible titles; counts of exclusions apply only to assessed titles. Targets without selected-history authors are excluded. One timing per query/method; excludes indexing and initialization.')
    (OUTPUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
