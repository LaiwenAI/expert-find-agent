"""Certify an ordered expert–source output under independent score intervals."""


def stable_witnesses(scores, owners, *, radius, cutoff, top):
    """Whether every score assignment in the intervals emits the same witnesses.

    Keys are original source positions, used to break ties. An owner list
    follows the declaration's fixed order. A repeated owner emits only its
    first applicable source. Empty output is stable if no source can emit.
    """
    pending = sorted((source for source in scores if owners[source]),
                     key=lambda source: (-scores[source], source))
    seen = set()
    for i, source in enumerate(pending):
        if all(owner in seen for owner in owners[source]):
            continue
        score = scores[source]
        if score + radius <= cutoff:
            return True
        lower = score - radius
        if lower <= cutoff:
            return False
        for rival in pending[i + 1:]:
            upper = scores[rival] + radius
            if (-upper, rival) >= (-lower, source):
                # Later upper bounds only decrease, with the same tie order.
                break
            if any(owner not in seen for owner in owners[rival]):
                return False
        seen.update(owners[source])
        if len(seen) >= top:
            return True
    return True
