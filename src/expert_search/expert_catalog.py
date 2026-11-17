"""Persistent ownership/offset catalog underlying expert evidence trees.

Metadata access optimization only; topic routing and applicability need evaluation.
"""
import hashlib
import itertools
import json
import sqlite3
from pathlib import Path


class ExpertCatalog:
    def __init__(self,corpus='artifacts/corpus',directory='artifacts/expert-catalog-v1'):
        self.source=Path(corpus)/'evidence.jsonl';root=Path(directory);root.mkdir(parents=True,exist_ok=True)
        hasher=hashlib.sha256()
        with self.source.open('rb') as stream:
            before=self._file_identity(stream.fileno())
            for chunk in iter(lambda:stream.read(1024*1024),b''):hasher.update(chunk)
            if self._file_identity(stream.fileno())!=before:raise ValueError('Evidence snapshot changed during initialization')
        if self._path_identity()!=before:raise ValueError('Evidence snapshot changed during initialization')
        self._source_identity=before
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
        database_stat=self.database.stat()
        self._database_identity=(database_stat.st_dev,database_stat.st_ino)
        self.db=sqlite3.connect(self.database.resolve().as_uri()+'?mode=ro',uri=True)
        self._data_version=self.db.execute('PRAGMA data_version').fetchone()[0]
        try:
            self._check_snapshot()
            self._verify_inventory()
        except BaseException:
            self.db.close()
            raise

    @staticmethod
    def _identity(stat):
        return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)

    @classmethod
    def _file_identity(cls,descriptor):
        import os
        return cls._identity(os.fstat(descriptor))

    def _path_identity(self):
        return self._identity(self.source.stat())

    def _check_snapshot(self):
        try:
            database_stat=self.database.stat()
            if ((database_stat.st_dev,database_stat.st_ino)!=self._database_identity or self._path_identity()!=self._source_identity or
                    self.db.execute('PRAGMA data_version').fetchone()[0]!=self._data_version):
                raise ValueError('Evidence catalog or source snapshot changed')
        except OSError as error:
            raise ValueError('Evidence source snapshot unavailable') from error

    def _verify_inventory(self):
        """Compare every catalog row with the current manifest-matched source.

        This detects stale, missing, injected or reassigned catalog entries,
        including edits made before this instance was opened. It does not
        authenticate the source data or an untrusted manifest itself.
        """
        self._check_snapshot()
        def source_rows(stream):
            while True:
                offset=stream.tell();body=stream.readline()
                if not body:break
                if not body.strip():continue
                row=json.loads(body)
                yield (row['id'],row['expert_id'],offset,len(body))
        with self.source.open('rb') as stream:
            if self._file_identity(stream.fileno())!=self._source_identity:raise ValueError('Evidence source snapshot changed')
            stored=self.db.execute('SELECT id,expert_id,offset,length FROM evidence ORDER BY offset')
            for original,entry in itertools.zip_longest(source_rows(stream),stored):
                if original!=entry:raise ValueError('Evidence catalog inventory differs from source')
            if self._file_identity(stream.fileno())!=self._source_identity:raise ValueError('Evidence source snapshot changed')
        self._check_snapshot()

    def _read_rows(self,rows):
        self._check_snapshot();result=[]
        with self.source.open('rb') as stream:
            if self._file_identity(stream.fileno())!=self._source_identity:raise ValueError('Evidence source snapshot changed')
            for sid,eid,offset,length in rows:
                stream.seek(offset);row=json.loads(stream.read(length))
                if row['id']!=sid or row['expert_id']!=eid:raise ValueError('Changed source offset or ownership')
                result.append(row)
            if self._file_identity(stream.fileno())!=self._source_identity:raise ValueError('Evidence source snapshot changed')
        self._check_snapshot()
        return result

    def for_evidence(self,evidence_ids):
        """Read only requested known sources; missing IDs remain absent.

        Returned records are checked against the frozen catalog offsets and
        owners. The caller must compare each record's owner with the intended
        expert; absence does not establish expert inability.
        """
        self._check_snapshot()
        ids=list(dict.fromkeys(evidence_ids))
        if any(not isinstance(s,str) or not s for s in ids):raise ValueError('Invalid evidence ID')
        if len(ids)>500:raise ValueError('Catalog read budget exceeds 500 sources')
        if not ids:return []
        placeholders=','.join('?' for _ in ids)
        rows=self.db.execute(f'SELECT id,expert_id,offset,length FROM evidence WHERE id IN ({placeholders}) ORDER BY offset',ids).fetchall()
        return self._read_rows(rows)

    def for_experts(self,expert_ids):
        self._check_snapshot()
        ids=sorted(set(expert_ids))
        if not ids:return []
        if len(ids)>500:raise ValueError('Catalog read budget exceeds 500 experts')
        placeholders=','.join('?' for _ in ids)
        rows=self.db.execute(f'SELECT id,expert_id,offset,length FROM evidence WHERE expert_id IN ({placeholders}) ORDER BY offset',ids).fetchall()
        return self._read_rows(rows)
