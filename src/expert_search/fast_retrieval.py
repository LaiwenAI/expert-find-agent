"""Exact BM25 engineering baseline with precomputed postings, not research novelty."""
import heapq
import json
import math
import sqlite3
from collections import defaultdict
from .retrieval import Search,tokens


def compile_index(source, target):
    src=sqlite3.connect(source); dst=sqlite3.connect(target)
    n,avg=src.execute('SELECT count(*),avg(length) FROM docs').fetchone()
    dst.executescript('CREATE TABLE docs(id INTEGER PRIMARY KEY, evidence_id TEXT, expert_id TEXT, body TEXT); CREATE TABLE scores(term TEXT, doc INTEGER, bm25 REAL, tfidf REAL);')
    for i,body in src.execute('SELECT id,body FROM docs'):
        r=json.loads(body);dst.execute('INSERT INTO docs VALUES(?,?,?,?)',(i,r['id'],r['expert_id'],body))
    df=dict(src.execute('SELECT term,count(*) FROM postings GROUP BY term'))
    batch=[]
    for term,doc,tf,length in src.execute('SELECT p.term,p.doc,p.tf,d.length FROM postings p JOIN docs d ON d.id=p.doc'):
        idf=math.log(1+(n-df[term]+.5)/(df[term]+.5))
        batch.append((term,doc,idf*tf*2.2/(tf+1.2*(.25+.75*length/avg)),idf*(1+math.log(tf))/math.sqrt(max(1,length))))
        if len(batch)>=10000:dst.executemany('INSERT INTO scores VALUES(?,?,?,?)',batch);batch=[]
    dst.executemany('INSERT INTO scores VALUES(?,?,?,?)',batch)
    dst.execute('CREATE INDEX term_idx ON scores(term)');dst.commit();dst.close();src.close()


class FastSearch(Search):
    def __init__(self, corpus, database):
        from pathlib import Path
        from .corpus import records
        self.db=sqlite3.connect(database)
        self.experts={r['id']:r for _,r in records(Path(corpus)/'experts.jsonl')}
        self.metadata={i:(ev,eid) for i,ev,eid in self.db.execute('SELECT id,evidence_id,expert_id FROM docs')}

    def evidence_for(self,query,expert_ids,n=3):
        """Apply one common lexical evidence policy to an external candidate pool."""
        if n<1:raise ValueError('Invalid evidence count')
        wanted=set(expert_ids)
        if wanted-set(self.experts):raise ValueError('Unknown expert IDs')
        allowed={doc for doc,(_,eid) in self.metadata.items() if eid in wanted}
        scores=defaultdict(float)
        for term in sorted(set(tokens(query))):
            for doc,score in self.db.execute('SELECT doc,bm25 FROM scores WHERE term=?',(term,)):
                if doc in allowed:scores[doc]+=score
        grouped=defaultdict(list)
        for doc in allowed:
            ev,eid=self.metadata[doc];grouped[eid].append((scores[doc],ev,doc))
        output={eid:[] for eid in wanted}
        for eid,hits in grouped.items():
            for value,_,doc in heapq.nsmallest(n,hits,key=lambda h:(-h[0],h[1])):
                row=json.loads(self.db.execute('SELECT body FROM docs WHERE id=?',(doc,)).fetchone()[0])
                output[eid].append(dict(row,score=value))
        return output

    def document_scores(self,query):
        """BM25 scores by unique evidence ID, without reading document bodies."""
        scores=defaultdict(float)
        for term in sorted(set(tokens(query))):
            for doc,score in self.db.execute('SELECT doc,bm25 FROM scores WHERE term=?',(term,)):
                scores[doc]+=score
        return {self.metadata[doc][0]:score for doc,score in scores.items()}

    def search(self, query, k=10, method='bm25-max', exclude=()):
        if k<1 or method not in ['bm25-max','bm25-vote','tfidf-max']:raise ValueError('Invalid k or method')
        excluded=set(exclude);scores=defaultdict(float)
        column='tfidf' if method=='tfidf-max' else 'bm25'
        for term in sorted(set(tokens(query))):
            for doc,score in self.db.execute(f'SELECT doc,{column} FROM scores WHERE term=?',(term,)):
                scores[doc]+=score
        grouped=defaultdict(list)
        for doc,score in scores.items():
            ev,eid=self.metadata[doc]
            if ev not in excluded:grouped[eid].append((score,ev,doc))
        ranked=[];hits_by_expert={}
        for eid,hits in grouped.items():
            top=heapq.nsmallest(5,hits,key=lambda h:(-h[0],h[1]));hits_by_expert[eid]=top
            value=sum(h[0] for h in top) if method=='bm25-vote' else top[0][0]
            ranked.append((value,eid))
        result=[]
        for score,eid in heapq.nsmallest(k,ranked,key=lambda x:(-x[0],x[1])):
            evidence=[]
            for value,_,doc in hits_by_expert[eid][:3]:
                row=json.loads(self.db.execute('SELECT body FROM docs WHERE id=?',(doc,)).fetchone()[0]);evidence.append(dict(row,score=value))
            result.append(dict(**self.experts[eid],score=score,evidence=evidence))
        return result
