"""Audit compiled BM25/TF-IDF postings against the source SQLite index."""
import argparse
import itertools
import math
import sqlite3
from pathlib import Path


def audit(source,compiled):
    source_uri=f'file:{Path(source).resolve()}?mode=ro'
    compiled_uri=f'file:{Path(compiled).resolve()}?mode=ro'
    with sqlite3.connect(source_uri,uri=True) as src,sqlite3.connect(compiled_uri,uri=True) as dst:
        n,avg=src.execute('SELECT count(*),avg(length) FROM docs').fetchone()
        if not n or not avg:raise ValueError('Empty or zero-length source index')
        df=dict(src.execute('SELECT term,count(*) FROM postings GROUP BY term'))
        left=src.execute('SELECT p.term,p.doc,p.tf,d.length FROM postings p JOIN docs d ON p.doc=d.id ORDER BY p.term,p.doc')
        right=dst.execute('SELECT term,doc,bm25,tfidf FROM scores ORDER BY term,doc')
        count=0;previous=None
        for a,b in itertools.zip_longest(left,right):
            if a is None or b is None:raise ValueError('Posting count mismatch')
            term,doc,tf,length=a
            if (term,doc)==previous:raise ValueError('Duplicate source posting')
            previous=(term,doc)
            if (term,doc)!=b[:2]:raise ValueError('Posting identity mismatch')
            if tf<=0 or length<=0:raise ValueError('Invalid source frequency or length')
            idf=math.log(1+(n-df[term]+.5)/(df[term]+.5))
            expected=(idf*tf*2.2/(tf+1.2*(.25+.75*length/avg)),idf*(1+math.log(tf))/math.sqrt(max(1,length)))
            if any(not math.isfinite(x) or not math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-15) for x,y in zip(b[2:],expected)):
                raise ValueError('Posting score mismatch')
            count+=1
        total=src.execute('SELECT count(*) FROM postings').fetchone()[0]
        if count!=total:raise ValueError('Source postings contain missing documents')
        return {'source_documents':n,'postings_verified':count,'terms':len(df),
                'scope':'Checks posting membership and precomputed scores against this source database, not semantic relevance or source-data correctness.'}


if __name__=='__main__':
    import json
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('compiled',type=Path)
    args=parser.parse_args();print(json.dumps(audit(args.source,args.compiled),indent=2))
