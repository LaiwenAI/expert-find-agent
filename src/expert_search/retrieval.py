"""Portable BM25 index. Chinese bigrams and English words share one vocabulary."""
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from .corpus import records


def tokens(text):
    parts = re.findall(r'[a-z0-9]+|[\u4e00-\u9fff]+', text.lower())
    return [t for p in parts for t in ([p] if not re.match(r'[\u4e00-\u9fff]',p) else ([p] if len(p)==1 else [p[i:i+2] for i in range(len(p)-1)]))]


def index(corpus, database):
    Path(database).parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(database)
    db.executescript('DROP TABLE IF EXISTS docs; DROP TABLE IF EXISTS postings; CREATE TABLE docs(id INTEGER PRIMARY KEY, body TEXT, length INTEGER); CREATE TABLE postings(term TEXT, doc INTEGER, tf INTEGER);')
    for i,(_,row) in enumerate(records(Path(corpus)/'evidence.jsonl')):
        counts=Counter(tokens(row['text']))
        db.execute('INSERT INTO docs VALUES(?,?,?)',(i,json.dumps(row,ensure_ascii=False),sum(counts.values())))
        db.executemany('INSERT INTO postings VALUES(?,?,?)',((t,i,n) for t,n in counts.items()))
    db.execute('CREATE INDEX term_index ON postings(term)')
    db.commit(); db.close()


class Search:
    def __init__(self, corpus, database):
        self.db=sqlite3.connect(database)
        self.experts={r['id']:r for _,r in records(Path(corpus)/'experts.jsonl')}
        self.n,self.avg=self.db.execute('SELECT count(*),avg(length) FROM docs').fetchone()
        if not self.n:
            raise ValueError('Empty evidence index')

    def search(self, query, k=10, method='bm25-max', exclude=()):
        if k<1: raise ValueError('k must be positive')
        scores=defaultdict(float)
        for term in set(tokens(query)):
            rows=self.db.execute('SELECT p.doc,p.tf,d.length FROM postings p JOIN docs d ON d.id=p.doc WHERE term=?',(term,)).fetchall()
            idf=math.log(1+(self.n-len(rows)+.5)/(len(rows)+.5))
            for doc,tf,length in rows:
                scores[doc]+=(idf*(1+math.log(tf))/math.sqrt(max(1,length)) if method=='tfidf-max' else idf*tf*2.2/(tf+1.2*(.25+.75*length/self.avg)))
        grouped=defaultdict(list)
        doc_ids=list(scores)
        for start in range(0,len(doc_ids),500):
            batch=doc_ids[start:start+500]
            placeholders=','.join('?' for _ in batch)
            for doc,body in self.db.execute(f'SELECT id,body FROM docs WHERE id IN ({placeholders})',batch):
                r=json.loads(body)
                if r['id'] not in exclude:
                    grouped[r['expert_id']].append((scores[doc],r))
        result=[]
        for eid, hits in grouped.items():
            hits.sort(key=lambda h:(-h[0],h[1]['id']))
            if method in ('bm25-max','tfidf-max'): score=hits[0][0]
            elif method=='bm25-vote': score=sum(s for s,_ in hits[:5])
            else: raise ValueError(f'Unknown retrieval method: {method}')
            result.append(dict(**self.experts[eid],score=score,evidence=[dict(r,score=s) for s,r in hits[:3]]))
        return sorted(result,key=lambda r:(-r['score'],r['id']))[:k]
