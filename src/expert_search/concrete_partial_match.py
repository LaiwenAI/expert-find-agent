"""Experimental focused matcher with concrete, nontruncated bridge instructions."""
from types import SimpleNamespace
from .focused_partial_match import FocusedPartialMethodMatcher

CONCRETE_INSTRUCTION='''For m/a/g use concise, COMPLETE Chinese sentences within the same character budgets; never clip a word or use ellipses to fit. For each d/t ground the central bridge in ONE documented own method, with citations that actually show that operation, not merely a title, motivation or prior work. In m state the source operation and the actual target operation and WHY their input/output or governing mechanism is compatible. In a state one specific adaptation or condition still requiring validation; "assume method X transfers to Y" is not a useful condition. Do not silently substitute a physical medium, signal representation, measured quantity, material or output meaning just because field or instrument names match. A plausible narrow component transfer remains valuable even when the source process/domain differs: state the preserved operation and necessary adaptation, rather than demanding prior realization of the whole enterprise system or future targets. Do not combine different papers into a previously achieved joint method. If no concrete connection can be expressed faithfully in the budget, use topic_only or insufficient as appropriate, never a clipped or generic positive claim. Missing full-system coverage is a gap, not by itself a reason to reject a valid partial method.'''


class ConcreteFocusedPartialMatcher(FocusedPartialMethodMatcher):
    def __init__(self,client):
        def create(**kwargs):
            messages=[dict(message) for message in kwargs['messages']]
            messages[0]['content']+='\n'+CONCRETE_INSTRUCTION
            return client.chat.completions.create(**dict(kwargs,messages=messages))
        super().__init__(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
