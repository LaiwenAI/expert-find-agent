"""Resident NumPy postings for the existing BM25 formula; no new ranker."""
from collections import defaultdict
import sqlite3
import numpy as np
from .retrieval import tokens


class ArrayBM25:
    def __init__(self, database):
        connection=sqlite3.connect(database)
        self.ids=tuple(row[0] for row in connection.execute('SELECT evidence_id FROM docs ORDER BY id'))
        positions={doc:i for i,(doc,) in enumerate(connection.execute('SELECT id FROM docs ORDER BY id'))}
        postings=defaultdict(list)
        for term,doc,value in connection.execute('SELECT term,doc,bm25 FROM scores ORDER BY rowid'):
            postings[term].append((positions[doc],value))
        connection.close()
        self.postings={term:(np.array([i for i,_ in rows],dtype=np.int32),np.array([v for _,v in rows],dtype=np.float64)) for term,rows in postings.items()}
        self.array_bytes=sum(i.nbytes+v.nbytes for i,v in self.postings.values())

    def document_scores(self,query):
        scores=np.zeros(len(self.ids),dtype=np.float64)
        active=np.zeros(len(self.ids),dtype=bool)
        for term in sorted(set(tokens(query))):
            posting=self.postings.get(term)
            if posting is not None:
                indices,values=posting
                # Compiled SQLite has exactly one posting per term/document.
                scores[indices]+=values;active[indices]=True
        return {self.ids[i]:float(scores[i]) for i in np.flatnonzero(active)}

    def profile_scores(self,query,profiles):
        """Unique-paper max/top3 aggregation, with the reference zero padding."""
        document_scores=self.document_scores(query)
        result={}
        for eid,papers in profiles.items():
            values=np.fromiter((document_scores.get(pid,0.) for pid in set(papers)),dtype=np.float64)
            if len(values)>3:values=np.partition(values,-3)[-3:]
            best=sorted((float(v) for v in values),reverse=True)
            result[eid]={'bm25-max':best[0] if best else 0.,'bm25-top3':sum(best)/3}
        return result
