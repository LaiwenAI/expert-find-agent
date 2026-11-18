"""Calibrate sparse document confirmation separately from contributor evidence."""

from collections import Counter
from difflib import SequenceMatcher
from itertools import product
import re


MODES = ("coverage", "ordered")
TIERS = ("short", "long")
SCORES = (.65, .75, .85, .90, .95, 1.)
MARGINS = (.05, .10, .15, .20, .30, .40, .50)
NOTICE = re.compile(r"^\s*(correction|retraction|expression of concern)\s*:", re.I)


def notice_kind(text):
    match = NOTICE.match(text)
    return match[1].casefold() if match else None


def coverage(query, segment):
    return sum((Counter(query) & Counter(segment)).values()) / len(query)


def ordered(query, segment):
    matcher = SequenceMatcher(None, query, segment, autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks()) / len(query)


def measure(query, shortlist, segments, mode):
    if not query or not shortlist:
        return None, 0., 0.
    score = coverage if mode == "coverage" else ordered
    ranks = [(max(score(query, segment) for segment in segments[doi]), doi) for doi in shortlist]
    ranks.sort(key=lambda entry: (-entry[0], entry[1]))
    best, doi = ranks[0]
    runner_up = ranks[1][0] if len(ranks) > 1 else 0.
    return doi, best, best - runner_up


def accepts(observation, policy):
    doi, score, gap = observation
    return (
        doi is not None and policy is not None
        and score >= policy["score"] and gap >= policy["margin"]
    )


def length_tier(query):
    return "short" if len(query) <= 8 else "long"


def calibrate(samples):
    feasible = []
    for threshold, margin in product(SCORES, MARGINS):
        policy = dict(score=threshold, margin=margin)
        accepted = [(observation, wanted) for observation, wanted in samples if accepts(observation, policy)]
        if accepted and all(observation[0] == wanted for observation, wanted in accepted):
            feasible.append((len(accepted), policy))
    if not feasible:
        return None
    return max(feasible, key=lambda item: (item[0], -item[1]["score"], -item[1]["margin"]))[1]
