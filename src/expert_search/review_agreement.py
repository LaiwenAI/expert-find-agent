"""Combine two blinded ordinal reviews without converting uncertainty to labels."""
from collections import Counter


def quadratic_kappa(pairs):
    """Quadratic weighted kappa for complete grade pairs; undefined if degenerate."""
    pairs=list(pairs)
    if any(type(a)!=int or type(b)!=int or not 0<=a<=3 or not 0<=b<=3 for a,b in pairs):
        raise ValueError('Require paired grades from 0 to 3')
    if not pairs:return None
    n=len(pairs);left=Counter(a for a,_ in pairs);right=Counter(b for _,b in pairs)
    observed=sum(((a-b)/3)**2 for a,b in pairs)/n
    expected=sum(left[a]*right[b]*((a-b)/3)**2 for a in range(4) for b in range(4))/(n*n)
    return 1-observed/expected if expected else None


def reconcile(first,second,adjudication=None):
    """Compare case-indexed validated review rows and retain unresolved cases.

    Adjudication can only resolve a pair of different completed grades. Reviewer
    IDs are checked for distinctness per case, not authenticated or verified as
    independent. All inputs must already pass packet/source validation.
    """
    if set(first)!=set(second) or (adjudication is not None and set(first)!=set(adjudication)):
        raise ValueError('Review inventories differ')
    outcomes={};pairs=[];status=Counter();owner_disagreements=0;owner_paired=0
    for case in sorted(first):
        a=first[case]['annotation'];b=second[case]['annotation'];ga=a['grade'];gb=b['grade']
        if ga is not None and gb is not None:
            if not all(isinstance(x,str) and x.strip() for x in (a['annotator'],b['annotator'])):raise ValueError('Missing reviewer')
            if a['annotator'].strip()==b['annotator'].strip():raise ValueError('Two distinct reviewer IDs required')
            pairs.append((ga,gb))
        if a['source_owner_verified'] is not None and b['source_owner_verified'] is not None:
            owner_paired+=1;owner_disagreements+=a['source_owner_verified']!=b['source_owner_verified']
        g3=adjudication[case]['annotation']['grade'] if adjudication is not None else None
        if g3 is not None:
            c=adjudication[case]['annotation'];name=c['annotator']
            if ga is None or gb is None or ga==gb:raise ValueError('Adjudication requires a disagreement')
            if not isinstance(name,str) or not name.strip() or name.strip() in (a['annotator'].strip(),b['annotator'].strip()):raise ValueError('Distinct adjudicator required')
            grade=g3;state='adjudicated'
        elif ga is None or gb is None:grade=None;state='missing_review'
        elif ga==gb:grade=ga;state='agreed'
        else:grade=None;state='disagreement'
        outcomes[case]={'grade':grade,'status':state};status[state]+=1
    return {'outcomes':outcomes,'counts':dict(status),'paired_grades':len(pairs),
            'exact_agreement':sum(a==b for a,b in pairs)/len(pairs) if pairs else None,
            'quadratic_weighted_kappa':quadratic_kappa(pairs),
            'paired_ownership_checks':owner_paired,'ownership_disagreements':owner_disagreements,
            'scope':'Structural reviewer-ID separation only; independence, qualifications, source ownership and entailment require external verification. Missing grades and unresolved disagreements remain unknown. Kappa is undefined with no expected disagreement.'}
