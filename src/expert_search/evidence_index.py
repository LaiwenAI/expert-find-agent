"""Source-preserving sentence index for constraint-specific evidence selection.
Segments are retrieval units, not verified semantic capability atoms.
"""
import json
import re
import sqlite3
from pathlib import Path
from .corpus import records
from .retrieval import tokens


def build(corpus='artifacts/corpus/evidence.jsonl', target='artifacts/evidence-segments.sqlite'):
    db=sqlite3.connect(target)
    db.executescript('CREATE TABLE segments(id INTEGER PRIMARY KEY, expert TEXT, body TEXT); CREATE VIRTUAL TABLE terms USING fts5(text); CREATE INDEX expert_idx ON segments(expert);')
    count=0
    for _,r in records(corpus):
        # Preserve original character offsets; split only at sentence or line boundaries.
        for match in re.finditer(r'[^\n。！？]+[。！？]?',r['text']):
            text=match.group()
            if len(text.strip())<12:continue
            row=dict(evidence_id=r['id'],expert_id=r['expert_id'],text=text,start=match.start(),end=match.end(),source=r['source'])
            count+=1
            db.execute('INSERT INTO segments VALUES(?,?,?)',(count,r['expert_id'],json.dumps(row,ensure_ascii=False)))
            db.execute('INSERT INTO terms(rowid,text) VALUES(?,?)',(count,' '.join(tokens(text))))
    db.commit();db.close();return count


class EvidenceIndex:
    def __init__(self,path='artifacts/evidence-segments.sqlite'):
        self.db=sqlite3.connect(path)

    def lookup(self,constraint,expert_ids,k=2):
        if not expert_ids or k<1:return {}
        terms=sorted(set(tokens(constraint)))
        if not terms:return {}
        expression=' OR '.join('"'+t+'"' for t in terms)
        slots=','.join('?' for _ in expert_ids)
        query=f'SELECT s.expert,s.body,bm25(terms) FROM terms JOIN segments s ON s.id=terms.rowid WHERE terms MATCH ? AND s.expert IN ({slots}) ORDER BY bm25(terms),s.id'
        result={eid:[] for eid in expert_ids}
        for eid,body,score in self.db.execute(query,[expression,*expert_ids]):
            if len(result[eid])<k:result[eid].append(dict(json.loads(body),retrieval_score=-score))
            if all(len(x)==k for x in result.values()):break
        return result
