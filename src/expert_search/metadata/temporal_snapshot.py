"""Build a conservative paper-only as-of corpus from a frozen evidence corpus."""
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path
from ..corpus import records
from .papers import availability, publication_interval


def build_temporal_snapshot(corpus, output, as_of):
    """Keep only papers provably published by the inclusive cutoff date.

    Unknown or overlapping date intervals and non-paper records are excluded;
    exclusion does not imply absent capability. Expert records are reduced to IDs because names and affiliations are not
    historically verified. Rebuild every retrieval index from this output.
    """
    if not isinstance(as_of, str): raise ValueError('Cutoff must be an ISO date')
    date.fromisoformat(as_of)
    corpus = Path(corpus); output = Path(output)
    if output.exists(): raise FileExistsError('Choose a new snapshot directory')
    experts_path = corpus / 'experts.jsonl'; evidence_path = corpus / 'evidence.jsonl'
    experts_bytes = experts_path.read_bytes(); source_bytes = evidence_path.read_bytes()
    experts = [row for _, row in records(experts_path)]
    expert_ids = {r['id'] for r in experts}
    if len(expert_ids) != len(experts): raise ValueError('Duplicate expert ID')
    kept = []; seen = set(); counts = Counter()
    for _, row in records(evidence_path):
        if row['id'] in seen or row['expert_id'] not in expert_ids: raise ValueError('Duplicate evidence or unknown owner')
        seen.add(row['id'])
        if row.get('kind') != 'paper':
            counts['non_paper_excluded'] += 1
            continue
        interval = row.get('publication')
        if not isinstance(interval, dict):
            counts['unknown_publication_date'] += 1
            continue
        canonical = publication_interval(interval.get('raw'))
        interval_fields = ('raw', 'precision', 'start', 'end_exclusive', 'status')
        if any(interval.get(key) != canonical[key] for key in interval_fields):
            counts['inconsistent_publication_interval'] += 1
            continue
        state = availability(canonical, as_of)
        counts[state] += 1
        if state == 'before_or_on_cutoff':
            allowed = ('id', 'expert_id', 'kind', 'text', 'title', 'url', 'source', 'publication', 'contribution_status', 'authorship_association')
            retained = {key: row[key] for key in allowed if key in row}
            retained['publication'] = canonical
            retained['contribution_status'] = 'unverified'
            retained['authorship_association'] = 'source_association_not_independently_verified'
            kept.append(retained)
    if not kept: raise ValueError('No evidence provably available by cutoff')
    evidence_bytes = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in kept).encode()
    output_experts_bytes = ''.join(json.dumps({'id': row['id']}, ensure_ascii=False) + '\n' for row in experts).encode()
    output.mkdir(parents=True, exist_ok=False)
    (output/'experts.jsonl').write_bytes(output_experts_bytes)
    (output/'evidence.jsonl').write_bytes(evidence_bytes)
    sha = lambda b: hashlib.sha256(b).hexdigest()
    report = dict(as_of=as_of, source_experts_sha256=sha(experts_bytes), source_evidence_sha256=sha(source_bytes),
                  output_experts_sha256=sha(output_experts_bytes), output_evidence_sha256=sha(evidence_bytes),
                  source_evidence=len(seen), kept_papers=len(kept), excluded=len(seen)-len(kept), counts=dict(counts),
                  scope='Conservative paper-only snapshot based on supplied publication-date intervals. Unknown, inconsistent and partial-overlap dates are excluded, not negatives. Source dates and author identity are not independently verified. Expert records retain IDs only; publication date is not proof of actual public availability. Paper fields are allowlisted to exclude citation counts and other extra metadata. Source text may still contain later edits or embedded metadata.')
    (output/'temporal-manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
