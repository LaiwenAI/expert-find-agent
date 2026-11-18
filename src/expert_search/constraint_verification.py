"""Resolve a ranked conjunction of hard constraints with source-bundled checks."""

from collections import Counter


def _rank(owners, observed, top, optimistic):
    result = []
    for expert, requirements in owners:
        citations = []
        for i, witnesses in enumerate(requirements):
            available = next((source for source in witnesses if (
                observed.get((expert, source), {}).get(i) is True
                or (optimistic and (expert, source) not in observed)
            )), None)
            if available is None:
                break
            citations.append(available)
        if len(citations) == len(requirements):
            result.append((expert, tuple(citations)))
            if len(result) == top:
                break
    return result


def resolve_constraints(owners, read, *, top, strategy="certificate"):
    """Read source declarations until the ranked experts and first witnesses are determined.

    `owners` orders experts by a fixed prior and supplies each criterion's ordered
    source witnesses. `read(expert, source)` maps criterion ordinals to True, False,
    or None. A read None is unavailable evidence and cannot establish that criterion.
    """
    if type(top) is not int or top < 1:
        raise ValueError("Expected a positive result count")
    if strategy not in {"exhaustive", "short_circuit", "bundled_first", "certificate"}:
        raise ValueError("Unknown constraint verification strategy")
    observed = {}
    possibilities = dict(owners)
    checks = 0

    def observe(expert, source):
        nonlocal checks
        observed[expert, source] = read(expert, source)
        checks += sum(source in witnesses for witnesses in possibilities[expert])

    if strategy == "exhaustive":
        for expert, requirements in owners:
            for source in dict.fromkeys(source for witnesses in requirements for source in witnesses):
                observe(expert, source)
    elif strategy in {"short_circuit", "bundled_first"}:
        confirmed = 0
        for expert, requirements in owners:
            if any(not witnesses for witnesses in requirements):
                continue
            ordered_checks = list(enumerate(requirements))
            if strategy == "bundled_first":
                share = Counter(source for sources in requirements for source in sources)
                ordered_checks.sort(key=lambda item: -share[item[1][0]])
            for i, witnesses in ordered_checks:
                for source in witnesses:
                    if (expert, source) not in observed:
                        observe(expert, source)
                    if observed[expert, source].get(i) is True:
                        break
                else:
                    break
            else:
                confirmed += 1
                if confirmed == top:
                    break
    else:
        while True:
            lower = _rank(owners, observed, top, False)
            upper = _rank(owners, observed, top, True)
            if lower == upper:
                break

            # The earliest optimistic expert still eligible to change the result
            # controls which declarations can establish or destroy its position.
            for expert, citations in upper:
                frontier = [source for source in citations if (expert, source) not in observed]
                if frontier:
                    benefit = Counter(frontier)
                    source = max(frontier, key=benefit.__getitem__)
                    observe(expert, source)
                    break
            else:
                raise AssertionError("Unequal bounds with no unresolved witness")
    return dict(ranking=_rank(owners, observed, top, False), reads=len(observed), checks=checks)
