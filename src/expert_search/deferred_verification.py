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


def resolve_shared_frontier(ordered, score, recheck, length, *, support, top, width, band, cost_aware):
    """Resolve only citations needed by the optimistic ranked output.

    The scorer is query-document based, so one verdict applies to all verified
    owners of a source. Outside-band verdicts have the same numerical premise
    as defer_and_rank. The length estimate affects scheduling only.
    """
    from collections import defaultdict

    owners = defaultdict(list)
    for expert, sources in ordered:
        for source in sources:
            owners[source].append(expert)
    definite = {}
    tentative = {}
    calls = 0
    while True:
        lower, _ = _extreme(ordered, definite, support, top, False)
        upper, _ = _extreme(ordered, definite, support, top, True)
        if lower == upper:
            return dict(selected=lower, checks=len(tentative), calls=calls,
                        unique_reads=len({source for _, source in tentative}), margins=tentative)

        pending = defaultdict(list)
        for position, (expert, citations) in enumerate(upper):
            for source in citations:
                if (expert, source) not in definite:
                    pending[source].append((position, expert))
        if not pending:
            raise AssertionError("Unequal ranked bounds without an unresolved citation")
        unseen = [source for source, positions in pending.items()
                  if (positions[0][1], source) not in tentative]
        choices = unseen or list(pending)
        if cost_aware:
            def priority(source):
                benefit = sum(1.0 / (rank + 1) for rank, _ in pending[source])
                return -benefit / length(source), pending[source][0][0], source

            choices.sort(key=priority)
        jobs = [(pending[source][0][1], source) for source in choices[:width]]
        verifier = score if unseen else recheck
        calls += 1
        for (_, source), value in zip(jobs, verifier(jobs), strict=True):
            for expert in owners[source]:
                key = (expert, source)
                tentative[key] = value
                if not unseen or abs(value) > band:
                    definite[key] = value
