"""Exact owner-max BM25 with source ordering deferred until the fused shortlist."""

import numpy as np

from .channel_fusion import fuse


class OwnedLexicalPool:
    def __init__(self, lexical, owners, source_ids):
        if len(owners) != len(source_ids) or len(lexical.ids) != len(source_ids):
            raise ValueError("Owner, source, and lexical rows must align")
        self.lexical = lexical
        self.source_ids = tuple(source_ids)
        self.owner_ids = tuple(sorted(set(owners)))
        owner_slot = {key: i for i, key in enumerate(self.owner_ids)}
        self.owner_rows = np.asarray([owner_slot[key] for key in owners], dtype=np.int32)
        rows = [[] for _ in self.owner_ids]
        for i in sorted(range(len(source_ids)), key=self.source_ids.__getitem__):
            rows[self.owner_rows[i]].append(i)
        self.by_owner = {
            owner: np.asarray(indices, dtype=np.int32)
            for owner, indices in zip(self.owner_ids, rows, strict=True)
        }

    @property
    def indexing_bytes(self):
        return self.owner_rows.nbytes + sum(rows.nbytes for rows in self.by_owner.values())

    def hybrid(self, text, dense_hits, *, k, source_depth, fusion_depth):
        scores, active = self.lexical.document_arrays(text)
        sparse_rows = np.flatnonzero(active)
        maximum = np.full(len(self.owner_ids), -np.inf)
        np.maximum.at(maximum, self.owner_rows[sparse_rows], scores[sparse_rows])
        positions = np.flatnonzero(np.isfinite(maximum))
        positions = positions[np.argsort(-maximum[positions], kind="stable")[:fusion_depth]]
        sparse = [self.owner_ids[i] for i in positions]
        dense = {entry["id"]: entry["evidence"] for entry in dense_hits}
        experts = fuse({"dense": list(dense), "lexical": sparse}, k=k, depth=fusion_depth)
        result = []
        for hit in experts:
            owner = hit["id"]
            evidence = self.by_owner.get(owner)
            sparse_ids = []
            if evidence is not None:
                candidates = evidence[active[evidence]]
                order = np.argsort(-scores[candidates], kind="stable")[:source_depth]
                sparse_ids = [self.source_ids[i] for i in candidates[order]]
            dense_ids = [entry["id"] for entry in dense.get(owner, ())]
            sources = fuse({"dense": dense_ids, "lexical": sparse_ids}, k=source_depth, depth=source_depth)
            result.append(dict(id=owner, evidence=sources))
        return result
