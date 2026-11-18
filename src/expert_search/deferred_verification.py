"""Defer tentative evidence decisions until they can affect ranked citations."""


def _extreme(ordered, margins, support, top, optimistic):
    confirmed, fallback = [], []
    for expert, sources in ordered:
        positives = []
        for source in sources:
            margin = margins.get((expert, source))
            if (margin is None and optimistic) or (margin is not None and margin >= 0.):
                positives.append(source)
        if len(positives) >= support:
            confirmed.append([expert, positives[:support]])
        else:
            fallback.append([expert, []])
    return (confirmed + fallback)[:top], len(confirmed)


def defer_and_rank(ordered, score, recheck, *, support, top, width, band):
    """Return stable ordered citations under all unresolved decisions in the band.

    Outside-band margins are trusted. The caller controls their numerical reliability.
    Both callbacks receive up to `width` distinct expert-source pairs in canonical order
    within an expert. Rechecked scores are treated as definitive.
    """
    definitive = {}
    observed = {}
    seen = {expert: 0 for expert, _ in ordered}
    rank = {expert: i for i, (expert, _) in enumerate(ordered)}
    calls = 0
    while True:
        lower, known_count = _extreme(ordered, definitive, support, top, False)
        upper, _ = _extreme(ordered, definitive, support, top, True)
        if lower == upper:
            return dict(selected=lower, checks=len(observed), calls=calls,
                        unique_reads=len({doc for _, doc in observed}), margins=observed)

        cutoff = rank[lower[-1][0]] if known_count >= top else len(ordered)
        next_jobs = []
        for i, (expert, sources) in enumerate(ordered):
            if i >= cutoff or seen[expert] == len(sources):
                continue
            values = [definitive.get((expert, source)) for source in sources]
            positives = sum(v is not None and v >= 0. for v in values)
            possible = positives + sum(v is None for v in values)
            if positives < support <= possible:
                next_jobs.append((expert, sources[seen[expert]]))
        if next_jobs:
            jobs = next_jobs[:width]
            calls += 1
            for pair, value in zip(jobs, score(jobs), strict=True):
                observed[pair] = value
                seen[pair[0]] += 1
                if abs(value) > band:
                    definitive[pair] = value
            continue

        # Fix only tentative citations in the currently optimistic output.
        jobs = [(expert, source) for expert, citations in upper for source in citations
                if (expert, source) in observed and (expert, source) not in definitive][:width]
        if not jobs:
            raise AssertionError("Outputs differ with no observable citation to resolve")
        for pair, value in zip(jobs, recheck(jobs), strict=True):
            definitive[pair] = observed[pair] = value
