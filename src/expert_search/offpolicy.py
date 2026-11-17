"""Conservative diagnostics for discrete-action off-policy evaluation."""
import math


def evaluate(events, target):
    """Compute IPS and self-normalized IPS from fully observed exposures.

    Each event must provide an action, reward, and the *full* logged action
    probability distribution. ``target`` maps each context to its full
    target distribution. A positive target probability with zero logging
    probability is rejected even when that action was not sampled. This is
    an overlap check, not proof that propensities, rewards, or logs are true.
    Missing outcomes are not imputed and cause the estimate to be withheld.
    """
    if not isinstance(events,list) or not events or not isinstance(target,dict):
        raise ValueError('Require events and target distributions')
    def distribution(d):
        if not isinstance(d,dict) or not d or any(not isinstance(k,str) or not k or type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for k,v in d.items()):
            raise ValueError('Invalid action probabilities')
        if not math.isclose(sum(d.values()),1.0,rel_tol=0,abs_tol=1e-9):
            raise ValueError('Probabilities must sum to one')
        return d
    weighted=[];weights=[];ids=set()
    for row in events:
        if not isinstance(row,dict):raise ValueError('Invalid event')
        eid=row.get('event_id');context=row.get('context');action=row.get('arm');reward=row.get('outcome')
        if not isinstance(eid,str) or not eid or eid in ids or not isinstance(context,str) or not context:
            raise ValueError('Invalid or duplicate event')
        ids.add(eid)
        if reward is None:raise ValueError('Missing outcome; estimate withheld')
        if type(reward) not in (int,float) or not math.isfinite(reward):raise ValueError('Invalid outcome')
        logged=distribution(row.get('logging_probabilities'));wanted=distribution(target.get(context))
        if set(logged)!=set(wanted) or action not in logged or logged[action]<=0:
            raise ValueError('Action distributions disagree')
        if any(wanted[a]>0 and logged[a]==0 for a in wanted):
            raise ValueError('Target policy lacks logging support')
        weight=wanted[action]/logged[action]
        weights.append(weight);weighted.append(weight*reward)
    sw=sum(weights);squared=sum(w*w for w in weights)
    return {'events':len(events),'ips':sum(weighted)/len(events),
            'snips':sum(weighted)/sw if sw else None,
            'effective_sample_size':sw*sw/squared if squared else 0.0,
            'max_weight':max(weights),
            'notice':'Requires correct logged probabilities, action overlap, observed outcomes and appropriate causal assumptions; no uncertainty interval or bias correction is provided.'}
