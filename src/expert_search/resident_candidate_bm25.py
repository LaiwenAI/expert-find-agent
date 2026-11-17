"""Resident sorted postings for exact candidate-scoped BM25 scoring."""
import math
import sqlite3
import time
from pathlib import Path

import numpy as np

from .retrieval import tokens


class ResidentCandidateBM25:
    """Build a read-only in-memory view; caller must bind it to a fixed snapshot.

    This index is an execution alternative for already compiled BM25 scores.
    Construction reads all postings and must be reported separately. It rejects
    ambiguous source IDs and duplicate term/document postings.
    """
    def __init__(self, database):
        start = time.perf_counter()
        with sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True) as db:
            docs = list(db.execute('SELECT id,evidence_id FROM docs ORDER BY id'))
            if len(docs) > np.iinfo(np.int32).max:
                raise ValueError('Too many evidence documents')
            if len({eid for _, eid in docs}) != len(docs):
                raise ValueError('Ambiguous evidence IDs')
            self.ids = tuple(eid for _, eid in docs)
            self.evidence_positions = {eid: i for i, eid in enumerate(self.ids)}
            doc_positions = {doc: i for i, (doc, _) in enumerate(docs)}
            count = db.execute('SELECT count(*) FROM scores').fetchone()[0]
            indices = np.empty(count, dtype=np.int32); values = np.empty(count, dtype=np.float64)
            offsets = {}; current = None; first = 0; previous_doc = None; cursor = 0
            for term, doc, value in db.execute('SELECT term,doc,bm25 FROM scores ORDER BY term,doc'):
                if term != current:
                    if current is not None: offsets[current] = (first, cursor)
                    current, first, previous_doc = term, cursor, None
                if doc not in doc_positions or doc == previous_doc or not math.isfinite(value):
                    raise ValueError('Invalid or duplicate compiled posting')
                indices[cursor] = doc_positions[doc]; values[cursor] = value
                cursor += 1; previous_doc = doc
            if current is not None: offsets[current] = (first, cursor)
            if cursor != count: raise ValueError('Changed posting count')
        indices.flags.writeable = False; values.flags.writeable = False
        self.indices, self.values, self.offsets = indices, values, offsets
        self.array_bytes = indices.nbytes + values.nbytes
        self.initialization_seconds = time.perf_counter() - start

    def selected_scores(self, query, evidence_ids):
        """Return exact scores for unique selected IDs; absent matches are zero."""
        identifiers = list(dict.fromkeys(evidence_ids))
        if any(eid not in self.evidence_positions for eid in identifiers):
            raise ValueError('Unknown evidence ID')
        positions = np.array(sorted(self.evidence_positions[eid] for eid in identifiers), dtype=np.int32)
        totals = np.zeros(len(positions), dtype=np.float64)
        for term in sorted(set(tokens(query))):
            offsets = self.offsets.get(term)
            if offsets is None or not len(positions): continue
            lo, hi = offsets; postings = self.indices[lo:hi]
            locations = np.searchsorted(postings, positions)
            slots = np.flatnonzero(locations < len(postings))
            slots = slots[postings[locations[slots]] == positions[slots]]
            totals[slots] += self.values[lo + locations[slots]]
        score_by_position = dict(zip(positions.tolist(), totals.tolist()))
        return {eid: score_by_position[self.evidence_positions[eid]] for eid in identifiers}
