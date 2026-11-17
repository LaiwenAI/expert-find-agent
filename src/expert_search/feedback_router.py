"""Experimental feedback router with explicit cold-start and missing outcomes."""
import math


class FeedbackRouter:
    """Choose between fixed arms from observed, context-specific feedback.

    The caller supplies an incumbent fixed independently of this router and a
    scalar outcome for the arm actually used. Missing outcomes are not zeros.
    Minimum counts are a heuristic, not a statistical safety guarantee; this
    class neither explores nor claims unbiased estimates from selected data.
    """
    def __init__(self, arms, incumbent, min_observations=3):
        if not isinstance(arms, (tuple, list)) or len(arms) < 2 or any(not isinstance(a, str) or not a for a in arms) or len(set(arms)) != len(arms):
            raise ValueError('Require distinct nonempty arm names')
        if incumbent not in arms or type(min_observations) is not int or min_observations < 1:
            raise ValueError('Invalid incumbent or observation threshold')
        self.arms=tuple(arms);self.incumbent=incumbent;self.min_observations=min_observations
        self._values={}

    def observe(self, context, chosen_arm, outcome):
        if not isinstance(context,str) or not context or chosen_arm not in self.arms:
            raise ValueError('Invalid context or chosen arm')
        if outcome is None:
            return
        if type(outcome) not in (int,float) or not math.isfinite(outcome):
            raise ValueError('Outcome must be finite or missing')
        values=self._values.setdefault(context,{a:[] for a in self.arms})
        values[chosen_arm].append(float(outcome))

    def decision(self, context):
        if not isinstance(context,str) or not context:
            raise ValueError('Invalid context')
        values=self._values.get(context,{a:[] for a in self.arms})
        counts={a:len(values[a]) for a in self.arms}
        if any(counts[a]<self.min_observations for a in self.arms):
            return {'arm':self.incumbent,'status':'insufficient_feedback','counts':counts}
        means={a:sum(values[a])/counts[a] for a in self.arms}
        best=max(means.values())
        arm=self.incumbent if means[self.incumbent]==best else next(a for a in self.arms if means[a]==best)
        return {'arm':arm,'status':'empirical_choice','counts':counts,'means':means}
