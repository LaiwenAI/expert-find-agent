"""Extract a primary affirmative activity for single-duty routing experiments."""

import re


CONTEXT_SUFFIX = re.compile(r"\s+\b(?:rather than|without|while|before)\b\s+", re.IGNORECASE)
NEGATIVE_OR_CONDITIONAL = re.compile(
    r"\b(?:no|not|never|cannot|can't|couldn't|wouldn't|didn't|doesn't|don't|if|unless|only)\b", re.IGNORECASE,
)


def affirmative_clause(phrase):
    """Project a trailing contrast or context clause without changing the asserted lead."""
    marker = CONTEXT_SUFFIX.search(phrase)
    if marker is None:
        return phrase
    lead = phrase[:marker.start()].rstrip(" ,;")
    if len(lead.split()) < 3 or NEGATIVE_OR_CONDITIONAL.search(lead):
        return phrase
    return lead
