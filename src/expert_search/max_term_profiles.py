"""Lexical late interaction: sum of per-term maximum document BM25 evidence.

An exploratory classical aggregation, not evidence of relational capability.
"""
from collections import defaultdict
import sqlite3
import numpy as np
from .retrieval import tokens


class MaxTermProfiles:
    def __init__(self,database,profiles):
        self.experts=tuple(sorted(profiles));owners=defaultdict(list)
        for i,eid in enumerate(self.experts):
            for pid in set(profiles[eid]):owners[pid].append(i)
        connection=sqlite3.connect(database)
        documents={doc:owners[evidence] for doc,evidence in connection.execute('SELECT id,evidence_id FROM docs')}
        postings=defaultdict(dict)
        for term,doc,value in connection.execute('SELECT term,doc,bm25 FROM scores'):
            values=postings[term]
            for owner in documents[doc]:
                if value>values.get(owner,0.):values[owner]=value
        connection.close()
        self.postings={term:(np.array(list(values),dtype=np.int32),np.array(list(values.values()),dtype=np.float64)) for term,values in postings.items()}
        self.array_bytes=sum(i.nbytes+v.nbytes for i,v in self.postings.values())

    def score(self,query):
        result=np.zeros(len(self.experts),dtype=np.float64)
        for term in sorted(set(tokens(query))):
            if term in self.postings:
                indices,values=self.postings[term];result[indices]+=values
        return {eid:float(result[i]) for i,eid in enumerate(self.experts)}
