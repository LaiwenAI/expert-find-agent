"""Validate an independently completed copy of a blinded review packet."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result = {r['snapshot_id']: r for r in rows}
    if len(rows) != len(result): raise ValueError('Duplicate case ID')
    return result


def check(root, review):
    root = Path(root); review = Path(review); manifest = json.loads((root / 'manifest.json').read_text())
    baseline = root / 'blind-review.jsonl'; mapping = root / 'private-mapping.json'
    if sha(baseline) != manifest['blind_sha256'] or sha(mapping) != manifest['mapping_sha256']:
        raise ValueError('Changed blinded packet or mapping')
    original, judged = read(baseline), read(review)
    if original.keys() != judged.keys() or len(original) != manifest['cases']: raise ValueError('Changed case inventory')
    counts = Counter()
    for case, row in judged.items():
        old = original[case]
        if row.keys() != old.keys() or any(row[k] != old[k] for k in old if k != 'annotation'):
            raise ValueError('Changed source or review instructions')
        note = row['annotation']
        if not isinstance(note, dict) or note.keys() != old['annotation'].keys(): raise ValueError('Changed annotation fields')
        grade = note['grade']
        if grade is not None and (type(grade) is not int or grade not in (0, 1, 2, 3)): raise ValueError('Invalid grade')
        if note['source_owner_verified'] is not None and type(note['source_owner_verified']) is not bool:
            raise ValueError('Invalid ownership status')
        if grade is not None and (not isinstance(note['annotator'], str) or not note['annotator'].strip()):
            raise ValueError('Missing annotator')
        if grade is not None and grade >= 2:
            citations = note['source_citations']; allowed = {source['id'] for source in row['source_view']}
            if not isinstance(citations, list) or not citations or any(not isinstance(x, str) or x not in allowed for x in citations):
                raise ValueError('Positive grade requires listed source IDs')
            for field in ('covered_operation', 'mechanism', 'open_assumptions'):
                if not isinstance(note[field], str) or not note[field].strip(): raise ValueError('Positive grade requires rationale')
        counts[str(grade) if grade is not None else 'missing'] += 1
    return {'cases': len(judged), 'grades': dict(sorted(counts.items())), 'review_sha256': sha(review),
            'scope': 'Structural validation only; does not verify reviewer identity, citation entailment, personal contribution or correctness. Missing remains missing; selected cases are not representative. Source and mapping remain private.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('root', type=Path); parser.add_argument('review', type=Path)
    args = parser.parse_args(); print(json.dumps(check(args.root, args.review), indent=2))
