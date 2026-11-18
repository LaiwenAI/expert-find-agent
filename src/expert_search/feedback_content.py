"""Experimental, source-anchored structured feedback; not a language parser."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Correction:
    expert_id: str
    query_phrase: str
    source_id: str
    quote: str
    action: str


def apply_corrections(query, ranking, corrections, owned_sources):
    """Stable promotion/demotion from exact query phrases and owned source quotes.

    ``owned_sources`` maps expert IDs to source-ID/text mappings. Invalid or
    ambiguous corrections abstain; lexical inclusion is not entailment.
    """
    if not isinstance(query, str) or not isinstance(ranking, list) or len(set(ranking)) != len(ranking):
        raise ValueError('Invalid query or ranking')
    votes = {}
    audit = []
    for correction in corrections:
        if not isinstance(correction, Correction):
            raise ValueError('Expected Correction')
        fields = (correction.expert_id, correction.query_phrase, correction.source_id, correction.quote)
        if any(not isinstance(x, str) or not x.strip() for x in fields) or correction.action not in ('promote', 'demote'):
            audit.append('invalid'); continue
        source = owned_sources.get(correction.expert_id, {}).get(correction.source_id)
        if correction.expert_id not in ranking or not isinstance(source, str) or correction.quote not in source:
            audit.append('unanchored'); continue
        if correction.query_phrase.casefold() not in query.casefold():
            audit.append('out_of_scope'); continue
        votes.setdefault(correction.expert_id, set()).add(correction.action)
        audit.append('matched')
    def bucket(expert):
        actions = votes.get(expert, set())
        if actions == {'promote'}: return 0
        if actions == {'demote'}: return 2
        return 1
    return {'ranking': sorted(ranking, key=bucket), 'audit': audit,
            'conflicts': sorted(e for e, actions in votes.items() if len(actions) > 1)}
