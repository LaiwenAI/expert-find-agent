"""Scope-aware evidence checks. This is a protocol extension, not new metric theory."""
from .constraint_metrics import evidence_gain


def scoped_gain(requirements, judgments, scope_validity=None):
    """Score atomic capabilities and explicit relationships separately.

    Each requirement has weight, mandatory, kind (atomic/relation), and, for
    a relation, dependencies. A relation needs its own support judgment:
    support for every endpoint never implies support for their relationship.
    """
    if not requirements:raise ValueError('No requirements')
    if scope_validity is not None:
        if set(scope_validity)-set(requirements) or any(v not in ('preserved','lost','unknown') for v in scope_validity.values()):
            raise ValueError('Invalid scope validity judgments')
    dependencies={}
    for key,requirement in requirements.items():
        kind=requirement.get('kind')
        parents=requirement.get('dependencies',[])
        if kind not in ('atomic','relation') or not isinstance(parents,list):
            raise ValueError('Invalid requirement kind or dependencies')
        if kind=='atomic' and parents:raise ValueError('Atomic requirement has dependencies')
        if kind=='relation' and (len(parents)<2 or len(set(parents))!=len(parents)):
            raise ValueError('A relation requires distinct endpoints')
        if any(p not in requirements or p==key for p in parents):
            raise ValueError('Unknown or self dependency')
        dependencies[key]=parents
    if set(judgments)-set(requirements):raise ValueError('Unknown judgment requirement')
    resolved={};visiting=set()
    def resolve(key):
        if key in resolved:return resolved[key]
        if key in visiting:raise ValueError('Cyclic requirement graph')
        visiting.add(key)
        label=judgments.get(key,'unknown')
        if label not in ('supported','contradicted','unknown'):raise ValueError('Invalid label')
        if label=='supported' and scope_validity is not None and scope_validity.get(key,'unknown')!='preserved':
            label='unknown'
        parents=[resolve(p) for p in dependencies[key]]
        # Incomplete prerequisite evidence blocks a claimed supported relation.
        # A contradiction about the relation remains an explicit contradiction.
        resolved[key]='unknown' if label=='supported' and any(p!='supported' for p in parents) else label
        visiting.remove(key)
        return resolved[key]
    for key in requirements:resolve(key)
    gain=evidence_gain(requirements,resolved)
    coverage={}
    for kind in ('atomic','relation'):
        subset={k:r for k,r in requirements.items() if r['kind']==kind}
        coverage[kind]=evidence_gain(subset,{k:resolved[k] for k in subset})['supported_coverage'] if subset else None
    return dict(**gain,atomic_coverage=coverage['atomic'],relation_coverage=coverage['relation'],
                resolved_judgments=resolved,
                warning='Separate externally judged relationship evidence is required; graph structure alone does not prove semantic support.')
