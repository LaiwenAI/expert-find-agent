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


def add_selective_indexes(database):
    """Build optional candidate-first indexes offline on a compiled database."""
    with sqlite3.connect(database) as db:
        db.execute('CREATE INDEX IF NOT EXISTS docs_expert_idx ON docs(expert_id)')
        db.execute('CREATE INDEX IF NOT EXISTS scores_doc_term_idx ON scores(doc,term)')


class FastSearch(Search):
    def __init__(self, corpus, database):
        from pathlib import Path
        from .corpus import records
        self.db=sqlite3.connect(database)
        self.experts={r['id']:r for _,r in records(Path(corpus)/'experts.jsonl')}
        self.metadata={i:(ev,eid) for i,ev,eid in self.db.execute('SELECT id,evidence_id,expert_id FROM docs')}
        self.evidence_documents={}
        self.ambiguous_evidence=set()
        for doc,(evidence_id,_) in self.metadata.items():
            if evidence_id in self.evidence_documents:self.ambiguous_evidence.add(evidence_id)
            self.evidence_documents[evidence_id]=doc

    def _bodies(self, doc_ids):
        """Fetch only selected evidence, in batches below SQLite's variable limit."""
        ids=list(dict.fromkeys(doc_ids)); result={}
        for start in range(0,len(ids),500):
            batch=ids[start:start+500]
            placeholders=','.join('?' for _ in batch)
            for doc,body in self.db.execute(f'SELECT id,body FROM docs WHERE id IN ({placeholders})',batch):
                result[doc]=json.loads(body)
        if len(result)!=len(ids):raise ValueError('Compiled index is missing evidence')
        return result

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
        selected={eid:heapq.nsmallest(n,hits,key=lambda h:(-h[0],h[1])) for eid,hits in grouped.items()}
        bodies=self._bodies(doc for hits in selected.values() for _,_,doc in hits)
        return {eid:[dict(bodies[doc],score=value) for value,_,doc in selected.get(eid,[])] for eid in wanted}

    def evidence_for_selective(self,query,expert_ids,n=3):
        """Retrieve the same lexical evidence using optional candidate-first indexes."""
        if n<1:raise ValueError('Invalid evidence count')
        wanted=set(expert_ids)
        if wanted-set(self.experts):raise ValueError('Unknown expert IDs')
        terms=sorted(set(tokens(query)));grouped=defaultdict(list);scores=defaultdict(float)
        identifiers=sorted(wanted)
        for start in range(0,len(identifiers),400):
            batch=identifiers[start:start+400];params=','.join('?' for _ in batch)
            docs=list(self.db.execute(f'SELECT id,evidence_id,expert_id FROM docs INDEXED BY docs_expert_idx WHERE expert_id IN ({params})',batch))
            for doc,ev,eid in docs:grouped[eid].append((doc,ev))
            for offset in range(0,len(terms),400):
                chunk=terms[offset:offset+400];term_params=','.join('?' for _ in chunk)
                sql=(f'SELECT s.doc,s.bm25 FROM docs d INDEXED BY docs_expert_idx '
                     f'CROSS JOIN scores s INDEXED BY scores_doc_term_idx '
                     f'WHERE d.expert_id IN ({params}) AND s.doc=d.id AND s.term IN ({term_params}) ORDER BY s.term,s.doc')
                for doc,value in self.db.execute(sql,batch+chunk):scores[doc]+=value
        selected={eid:heapq.nsmallest(n,[(scores[doc],ev,doc) for doc,ev in hits],key=lambda h:(-h[0],h[1])) for eid,hits in grouped.items()}
        bodies=self._bodies(doc for hits in selected.values() for _,_,doc in hits)
        return {eid:[dict(bodies[doc],score=value) for value,_,doc in selected.get(eid,[])] for eid in wanted}

    def evidence_from_scores(self, expert_ids, evidence_by_expert, scores, n=3):
        """Materialize owned evidence using already-computed document scores.

        Every source for a requested expert must be present in the score map;
        a missing score is not silently treated as zero. IDs must be unique and
        owned by the expert in this compiled snapshot.
        """
        if type(n) is not int or n < 1:
            raise ValueError('Invalid evidence count')
        wanted = set(expert_ids)
        if wanted - set(self.experts):
            raise ValueError('Unknown expert IDs')
        selected = {}
        for expert in wanted:
            hits = []
            seen = set()
            for evidence_id in evidence_by_expert.get(expert, []):
                if evidence_id in seen or evidence_id in self.ambiguous_evidence or evidence_id not in self.evidence_documents or evidence_id not in scores:
                    raise ValueError('Missing, duplicate or ambiguous evidence score')
                seen.add(evidence_id)
                doc = self.evidence_documents[evidence_id]
                if self.metadata[doc][1] != expert:
                    raise ValueError('Evidence owner mismatch')
                value = scores[evidence_id]
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError('Invalid evidence score')
                hits.append((value, evidence_id, doc))
            selected[expert] = heapq.nsmallest(n, hits, key=lambda h: (-h[0], h[1]))
        bodies = self._bodies(doc for hits in selected.values() for _, _, doc in hits)
        return {expert: [dict(bodies[doc], score=value) for value, _, doc in selected[expert]] for expert in wanted}

    def selective_document_scores(self,query,evidence_ids):
        """Exact BM25 scores for selected evidence IDs, without reading bodies.

        Requires the optional doc/term index; unscored documents receive zero.
        Duplicate corpus IDs are rejected when requested because a source ID
        would otherwise refer to an ambiguous document.
        """
        identifiers=list(dict.fromkeys(evidence_ids))
        if any(eid not in self.evidence_documents or eid in self.ambiguous_evidence for eid in identifiers):
            raise ValueError('Unknown or ambiguous evidence ID')
        terms=sorted(set(tokens(query)));scores=defaultdict(float)
        docs=sorted(self.evidence_documents[eid] for eid in identifiers)
        for start in range(0,len(docs),400):
            batch=docs[start:start+400];params=','.join('?' for _ in batch)
            for offset in range(0,len(terms),400):
                chunk=terms[offset:offset+400];term_params=','.join('?' for _ in chunk)
                sql=(f'SELECT doc,bm25 FROM scores INDEXED BY scores_doc_term_idx '
                     f'WHERE doc IN ({params}) AND term IN ({term_params}) ORDER BY term,doc')
                for doc,value in self.db.execute(sql,batch+chunk):scores[doc]+=value
        return {eid:scores[self.evidence_documents[eid]] for eid in identifiers}

    def prepare_adaptive_scoring(self):
        """Cache posting counts for a simple cost-based scoring route."""
        self.term_posting_counts=dict(self.db.execute('SELECT term,count(*) FROM scores GROUP BY term'))

    def adaptive_document_scores(self,query,evidence_ids):
        """Return exact selected BM25 scores using a posting-count cost heuristic.

        The heuristic does not guarantee speed. Initialization can be made
        explicit with ``prepare_adaptive_scoring`` and must be timed separately.
        """
        identifiers=list(dict.fromkeys(evidence_ids))
        if any(eid not in self.evidence_documents or eid in self.ambiguous_evidence for eid in identifiers):
            raise ValueError('Unknown or ambiguous evidence ID')
        if not hasattr(self,'term_posting_counts'):self.prepare_adaptive_scoring()
        terms=set(tokens(query))
        active=sum(t in self.term_posting_counts for t in terms)
        global_work=sum(self.term_posting_counts.get(t,0) for t in terms)
        selective_work=len(identifiers)*active
        if selective_work < global_work:
            scores=self.selective_document_scores(query,identifiers);route='selective'
        else:
            full=self.document_scores(query);scores={eid:full.get(eid,0.) for eid in identifiers};route='global'
        return dict(scores=scores,route=route,estimated_global_postings=global_work,
                    estimated_selective_probes=selective_work)

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
        top_experts=heapq.nsmallest(k,ranked,key=lambda x:(-x[0],x[1]))
        bodies=self._bodies(doc for _,eid in top_experts for _,_,doc in hits_by_expert[eid][:3])
        for score,eid in top_experts:
            evidence=[]
            for value,_,doc in hits_by_expert[eid][:3]:
                evidence.append(dict(bodies[doc],score=value))
            result.append(dict(**self.experts[eid],score=score,evidence=evidence))
        return result
