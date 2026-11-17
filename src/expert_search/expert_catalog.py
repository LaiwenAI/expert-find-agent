"""Persistent ownership/offset catalog underlying expert evidence trees.

Metadata access optimization only; topic routing and applicability need evaluation.
"""
import hashlib
import json
import sqlite3
from pathlib import Path


class ExpertCatalog:
    def __init__(self,corpus='artifacts/corpus',directory='artifacts/expert-catalog-v1'):
        self.source=Path(corpus)/'evidence.jsonl';root=Path(directory);root.mkdir(parents=True,exist_ok=True)
        hasher=hashlib.sha256()
        with self.source.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):hasher.update(chunk)
        digest=hasher.hexdigest()
        self.database=root/'catalog.sqlite';marker=root/'manifest.json'
        if marker.exists():
            spec=json.loads(marker.read_text())
            if spec['source_sha256']!=digest:raise ValueError('Changed expert evidence snapshot')
        else:
            if self.database.exists():raise ValueError('Incomplete catalog; inspect before rebuilding')
            db=sqlite3.connect(self.database);db.execute('CREATE TABLE evidence(id TEXT PRIMARY KEY,expert_id TEXT,offset INTEGER,length INTEGER)')
            count=0
            with self.source.open('rb') as stream:
                while True:
                    offset=stream.tell();body=stream.readline()
                    if not body:break
                    if not body.strip():continue
                    row=json.loads(body);db.execute('INSERT INTO evidence VALUES(?,?,?,?)',(row['id'],row['expert_id'],offset,len(body)));count+=1
            db.execute('CREATE INDEX owner_idx ON evidence(expert_id)');db.commit();db.close()
            marker.write_text(json.dumps(dict(version=1,source_sha256=digest,records=count),indent=2))
        self.db=sqlite3.connect(self.database)

    def for_evidence(self,evidence_ids):
        """Read only requested known sources; missing IDs remain absent.

        Returned records are checked against the frozen catalog offsets and
        owners. The caller must compare each record's owner with the intended
        expert; absence does not establish expert inability.
        """
        ids=list(dict.fromkeys(evidence_ids))
        if any(not isinstance(s,str) or not s for s in ids):raise ValueError('Invalid evidence ID')
        if len(ids)>500:raise ValueError('Catalog read budget exceeds 500 sources')
        if not ids:return []
        placeholders=','.join('?' for _ in ids)
        rows=self.db.execute(f'SELECT id,expert_id,offset,length FROM evidence WHERE id IN ({placeholders}) ORDER BY offset',ids).fetchall()
        result=[]
        with self.source.open('rb') as stream:
            for sid,eid,offset,length in rows:
                stream.seek(offset);row=json.loads(stream.read(length))
                if row['id']!=sid or row['expert_id']!=eid:raise ValueError('Changed source offset or ownership')
                result.append(row)
        return result

    def for_experts(self,expert_ids):
        ids=sorted(set(expert_ids))
        if not ids:return []
        if len(ids)>500:raise ValueError('Catalog read budget exceeds 500 experts')
        placeholders=','.join('?' for _ in ids)
        rows=self.db.execute(f'SELECT id,expert_id,offset,length FROM evidence WHERE expert_id IN ({placeholders}) ORDER BY offset',ids).fetchall()
        result=[]
        with self.source.open('rb') as stream:
            for sid,eid,offset,length in rows:
                stream.seek(offset);row=json.loads(stream.read(length))
                if row['id']!=sid or row['expert_id']!=eid:raise ValueError('Changed source offset or ownership')
                result.append(row)
        return result
