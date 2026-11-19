"""Conservative capability postings over owned, version-pinned declarations."""

from collections import defaultdict


def _positions(mask):
    while mask:
        lowest = mask & -mask
        yield lowest.bit_length() - 1
        mask ^= lowest


class RoleEvidenceIndex:
    """Index confirmed role names while keeping opaque records eligible for verification.

    Owners are in the frozen prior order. Each owner's iterable contains
    (source, roles) records in canonical citation order. A None roles value
    denotes an opaque declaration, whereas an empty collection is a verified
    negative for every role. The caller owns provenance and visibility checks.
    """

    def __init__(self, owners):
        self.owners = tuple(owners)
        self.documents = []
        self.positive = defaultdict(int)
        self.opaque = 0
        seen_owners = set()
        for i, (owner, documents) in enumerate(self.owners):
            if owner in seen_owners:
                raise ValueError("Duplicate owner")
            seen_owners.add(owner)
            records = tuple((source, None if roles is None else frozenset(roles))
                            for source, roles in documents)
            if len({source for source, _ in records}) != len(records):
                raise ValueError("Duplicate witness for one owner")
            self.documents.append(records)
            if any(roles is None for _, roles in records):
                self.opaque |= 1 << i
            for _, roles in records:
                if roles is not None:
                    for role in roles:
                        self.positive[role] |= 1 << i
        self.owners = tuple(owner for owner, _ in self.owners)

    def rank(self, requirements, read, *, top, postings="auto"):
        """Return the fixed-prior top experts and canonical cited source per role."""
        requirements = tuple(requirements)
        if type(top) is not int or top < 1 or not requirements:
            raise ValueError("Expected a positive cutoff and at least one requirement")
        if postings not in (True, False, "auto"):
            raise ValueError("Expected forced postings, forward scanning, or automatic selection")
        size = len(self.owners)
        all_owners = (1 << size) - 1
        possible = all_owners
        if postings:
            for role in requirements:
                possible &= self.positive.get(role, 0) | self.opaque
        if postings == "auto" and possible.bit_count() * 2 > size:
            possible = all_owners

        ranking = []
        reads = candidates = 0
        for i in range(size) if possible == all_owners else _positions(possible):
            candidates += 1
            owner = self.owners[i]
            documents = self.documents[i]
            observed = {}
            citations = []
            for role in requirements:
                for source, roles in documents:
                    if roles is None:
                        if source not in observed:
                            observed[source] = frozenset(read(owner, source) or ())
                            reads += 1
                        roles = observed[source]
                    if role in roles:
                        citations.append(source)
                        break
                else:
                    break
            else:
                ranking.append((owner, tuple(citations)))
                if len(ranking) == top:
                    break
        return dict(ranking=ranking, reads=reads, visited_candidates=candidates)
