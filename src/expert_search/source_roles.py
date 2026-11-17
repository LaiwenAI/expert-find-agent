"""Conservative source-role normalization; never creates source evidence."""
ROLES = ('method', 'result', 'interpretation', 'background', 'other', 'uncertain')


def normalize_roles(value, size):
    """Conflicting/missing annotations become uncertain; invalid IDs fail closed.

    This validates reference structure, not whether an assigned role is true.
    Repeated IDs within one role carry no additional vote.
    """
    if type(size) is not int or size < 0:
        raise ValueError('Invalid source inventory size')
    if not isinstance(value, dict) or set(value) != set(ROLES):
        raise ValueError('Invalid source role inventory')
    assignments = [set() for _ in range(size)]
    duplicates = 0
    for role in ROLES:
        group = value[role]
        if not isinstance(group, list) or any(type(i) is not int or not 0 <= i < size for i in group):
            raise ValueError('Foreign or invalid source ID')
        duplicates += len(group) - len(set(group))
        for i in group:
            assignments[i].add(role)
    result = {role: [] for role in ROLES}
    missing = conflicts = 0
    for i, labels in enumerate(assignments):
        if not labels:
            missing += 1
        elif len(labels) > 1:
            conflicts += 1
        role = next(iter(labels)) if len(labels) == 1 else 'uncertain'
        result[role].append(i)
    return dict(roles=result, missing=missing, conflicts=conflicts,
                within_role_duplicates=duplicates,
                notice='Uncertain means unresolved source role, not absent expertise. Structural normalization does not validate semantic attribution.')


def prioritize_roles(ranking, roles):
    """Stable role preference, retaining every ranked source, without new scores."""
    if len(set(ranking)) != len(ranking):
        raise ValueError('Duplicate source ranking')
    preferred = set().union(*(set(roles[r]) for r in ROLES[:3]))
    return sorted(ranking, key=lambda i: i not in preferred)
