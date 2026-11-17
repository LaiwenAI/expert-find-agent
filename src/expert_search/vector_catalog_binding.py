"""Validate a frozen source/vector manifest against an owned evidence catalog."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .owned_vector_index import ExactOwnedVectorIndex
from .gpu_owned_vector_index import GpuOwnedVectorIndex


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def identity(path):
    st=Path(path).stat();return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)


class VectorCatalogBinding:
    """Bind a fixed index to one catalog and previously recorded build inputs.

    Expected file hashes must come from an independently preserved encoding
    run. Every branch prefix and owner is checked against the catalog, and
    loaded vectors are checked against the frozen vector file. This cannot prove
    that the encoder produced those vectors or that a source is truthful.
    Index vectors, ownership metadata, graph and GPU snapshot must not be
    mutated after binding; that caller contract is not rehashed per query.
    """
    def __init__(self,index,catalog,branches_path,vectors_path,*,branches_sha256,vectors_sha256,prefix_chars=1200):
        if type(prefix_chars) is not int or prefix_chars<1:raise ValueError('Invalid prefix limit')
        if not isinstance(index,(ExactOwnedVectorIndex,GpuOwnedVectorIndex)):
            raise ValueError('Require an exact owned vector index or GPU scorer')
        reference=index.reference if isinstance(index,GpuOwnedVectorIndex) else index
        if not isinstance(reference,ExactOwnedVectorIndex) or not callable(getattr(catalog,'for_evidence',None)) or not callable(getattr(catalog,'_check_snapshot',None)):
            raise ValueError('Require an exact owned vector index and verified evidence catalog')
        branches_path=Path(branches_path);vectors_path=Path(vectors_path)
        before=(identity(branches_path),identity(vectors_path))
        if sha(branches_path)!=branches_sha256 or sha(vectors_path)!=vectors_sha256:raise ValueError('Changed frozen build inputs')
        rows=[json.loads(line) for line in branches_path.read_text().splitlines()]
        if len(rows)!=len(reference.source_ids) or len({r['id'] for r in rows})!=len(rows):raise ValueError('Changed source inventory')
        if [r['id'] for r in rows]!=reference.source_ids or [[r['expert_id']] for r in rows]!=reference.source_owners:
            raise ValueError('Vector index IDs or ownership differ from build records')
        frozen=np.load(vectors_path,mmap_mode='r',allow_pickle=False)
        if frozen.shape!=reference.vectors.shape or frozen.dtype!=reference.vectors.dtype or not np.array_equal(frozen,reference.vectors):
            raise ValueError('Index vectors differ from frozen build matrix')
        for start in range(0,len(rows),500):
            chunk=rows[start:start+500];actual={r['id']:r for r in catalog.for_evidence([r['id'] for r in chunk])}
            if len(actual)!=len(chunk):raise ValueError('Catalog missing indexed sources')
            for branch in chunk:
                doc=actual[branch['id']];prefix=doc['text'][:prefix_chars]
                if (doc['expert_id']!=branch['expert_id'] or prefix!=branch['text'] or
                        hashlib.sha256(prefix.encode()).hexdigest()!=branch['source_prefix_sha256']):
                    raise ValueError('Indexed source prefix or owner differs from catalog')
        catalog._check_snapshot()
        if (identity(branches_path),identity(vectors_path))!=before:raise ValueError('Build inputs changed during binding')
        self.index=index;self.catalog=catalog;self.branches_path=branches_path;self.vectors_path=vectors_path;self._identities=before
        self.branches_sha256=branches_sha256;self.vectors_sha256=vectors_sha256;self.sources=len(rows)

    def ensure_current(self,index,catalog):
        if index is not self.index or catalog is not self.catalog:raise ValueError('Binding belongs to different index or catalog')
        try:
            if (identity(self.branches_path),identity(self.vectors_path))!=self._identities:raise ValueError('Frozen build inputs changed')
        except OSError as error:raise ValueError('Frozen build inputs unavailable') from error
        catalog._check_snapshot()
