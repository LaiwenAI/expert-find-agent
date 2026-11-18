"""Compare a single quoted measurement to a quoted, compatible threshold."""

from dataclasses import dataclass
from decimal import Decimal
import operator
import re

from expert_search.bilingual.constraints import NUMBER

UNITS = {
    '%': ('ratio', Decimal('.01')), 'percent': ('ratio', Decimal('.01')),
    's': ('duration', Decimal(1)), 'ms': ('duration', Decimal('.001')),
    'K': ('temperature_delta', Decimal(1)), 'mK': ('temperature_delta', Decimal('.001')),
    'kB': ('storage', Decimal(1000)), 'MB': ('storage', Decimal(1000000)),
    'items/s': ('item_rate', Decimal(1)), 'requests/s': ('request_rate', Decimal(1)),
    'W': ('power', Decimal(1)),
}
UNIT = re.compile(r'\s*(' + '|'.join(map(re.escape, sorted(UNITS, key=len, reverse=True))) + r')(?![A-Za-z/])')
COMPARATORS = (
    (r'(?:at most|no greater than|no more than|≤|<=)\s*$', operator.le),
    (r'(?:at least|no lower than|no less than|≥|>=)\s*$', operator.ge),
    (r'(?:strictly below|strictly less than|<)\s*$', operator.lt),
    (r'(?:strictly above|strictly more than|>)\s*$', operator.gt),
)
RANGE = re.compile(r'(?:\bbetween\b|\bfrom\b)\s*\d|\d\s*[–—-]\s*\d|\d\s+to\s+\d', re.I)


@dataclass(frozen=True)
class Quantity:
    dimension: str
    value: Decimal
    start: int
    end: int


def quantities(text):
    results = []
    for match in NUMBER.finditer(text):
        unit = UNIT.match(text, match.end())
        if unit:
            dimension, factor = UNITS[unit.group(1)]
            reading = Decimal(match.group().replace(',', '').replace('−', '-'))
            results.append(Quantity(dimension, reading * factor, match.start(), unit.end()))
    return results


def constraint(request):
    matches = []
    for quantity in quantities(request):
        prefix = request[:quantity.start]
        for pattern, compare in COMPARATORS:
            if re.search(pattern, prefix, re.I):
                matches.append((quantity, compare))
                break
    return matches[0] if len(matches) == 1 else None


def evaluate_measurement(condition, evidence):
    if condition is None or RANGE.search(evidence):
        return 'unknown', None
    limit, compare = condition
    readings = [r for r in quantities(evidence) if r.dimension == limit.dimension]
    if len(readings) != 1:
        return 'unknown', None
    result = readings[0]
    return ('eligible' if compare(result.value, limit.value) else 'reject',
            evidence[result.start:result.end])
