"""Audit transfer eligibility without pre-normalized readings or bounds."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import re
import time

from expert_search.numeric_evidence import constraint, evaluate_measurement
from experiments.transfer.constraint_counterfactual import Judge, LABELS, OPS, RULE, generate, sha

SEED = 20261124
CONTROLS = ('related_measurement', 'proposed_target', 'ambiguous_range', 'incompatible_unit')
METHODS = ('direct_scope', 'whole_card', 'select_clause')
PARAPHRASE = {'<=': 'no greater than', '>=': 'no lower than', '<': '<', '>': '>'}
CLAUSES = re.compile(r';\s+|(?<=[.!?])\s+(?=[A-Z])')


def challenge():
    original = generate()
    stressed = []
    groups = defaultdict(list)
    rng = random.Random(SEED)
    for row in original:
        if row['view'] == 'intact':
            groups[row['pair']].append(row)
    for i, (key, pair) in enumerate(groups.items()):
        variant = CONTROLS[(i + i // 4) % len(CONTROLS)]
        report_first = rng.random() < .5
        for row in pair:
            source = row['gold_source']
            card = row['cards'][source]['text']
            bound_info = constraint(row['query'])
            quantity, _ = bound_info
            text = row['query'][quantity.start:quantity.end]
            unit = text.split()[-1]
            amount = int(row['cards'][source]['reading'])
            metric = row['scope']['metric']
            setting = row['scope']['setting']
            noise = f'The secondary diagnostic score in {setting} was {amount + 11} {unit}.'
            if variant == 'proposed_target':
                noise = f'The proposed target for {metric} in {setting} was {amount + 9} {unit}.'
            if variant in ('related_measurement', 'proposed_target'):
                order = (card, noise) if report_first else (noise, card)
                replacement = ' '.join(order)
                gold = row['gold']
            elif variant == 'ambiguous_range':
                replacement = f'In {setting} the reported {metric} ranged from {amount - 4} to {amount + 4} {unit}. No point measurement was reported.'
                gold = 'unknown'
            else:
                replacement = f'In {setting} the recorded {metric} was {amount} W.'
                gold = 'unknown'
            cards = [d['text'] for d in row['cards']]
            cards[source] = replacement
            query = row['query'].replace(OPS[row['operator']][0], PARAPHRASE[row['operator']], 1)
            stressed.append({'id': row['id'] + '-stress', 'pair': key + '-stress',
                             'view': variant, 'query': query, 'scope': row['scope'],
                             'cards': cards, 'gold': gold})
    clean = [{'id': r['id'], 'pair': r['pair'], 'view': r['view'], 'query': r['query'],
              'scope': r['scope'], 'cards': [d['text'] for d in r['cards']], 'gold': r['gold']}
             for r in original]
    return clean + stressed


def select_card(judge, scope, cards):
    picked, elapsed = judge(
        'Choose the test card that provides an empirical measurement of the requested metric '
        'in the exact provided setting. Design targets, other settings and unrelated metrics '
        'cannot qualify. Return only its zero-based digit 0–3, or 4 when none qualifies.',
        {'scope': scope, 'cards': cards}, (0, 1, 2, 3, None))
    if picked is None:
        return None, elapsed
    verified, verification = judge(
        'Does the card report an empirical result for the exact metric in the exact setting? '
        'An intended target or a result from another setting is not a measurement. '
        'Reply 0 for no or 1 for yes.',
        {'scope': scope, 'card': cards[picked]}, (False, True))
    return (picked if verified else None), elapsed + verification


def evaluate(rows, judge):
    output, cache, clauses_cache = [], {}, {}
    for row in rows:
        cards, scope = row['cards'], row['scope']
        operator_started = time.perf_counter()
        bound = constraint(row['query'])
        parsing = (time.perf_counter() - operator_started) * 1000
        raw, direct_ms = judge(
            RULE + ' 0 means eligible. 1 means reject. 2 means unknown. Return only one digit.',
            {'request': row['query'], 'scope': scope, 'cards': cards}, LABELS)
        key = row['pair'], tuple(cards)
        if key not in cache:
            cache[key] = select_card(judge, scope, cards)
        selected, retrieval_ms = cache[key]
        text = cards[selected] if selected is not None else ''
        tic = time.perf_counter()
        full, quote = evaluate_measurement(bound, text)
        full_ms = parsing + retrieval_ms + (time.perf_counter() - tic) * 1000
        clause, clause_quote, clause_ms = full, quote, full_ms
        snippets = CLAUSES.split(text)
        if full == 'unknown' and len(snippets) == 2:
            if key not in clauses_cache:
                clauses_cache[key] = judge(
                    'Find the exact clause that supplies a single empirical measurement of this metric '
                    'in this setting. Ignore proposed targets and other metrics. Reply 0 or 1 for the '
                    'corresponding clause, or 2 if neither or both qualify.',
                    {'scope': scope, 'clauses': snippets}, (0, 1, None))
            chosen, extra_ms = clauses_cache[key]
            tic = time.perf_counter()
            evidence = snippets[chosen] if chosen is not None else ''
            clause, clause_quote = evaluate_measurement(bound, evidence)
            clause_ms += extra_ms + (time.perf_counter() - tic) * 1000
        for value in (quote, clause_quote):
            if value is not None and value not in text:
                raise ValueError('Unanchored result')
        output.append({'id': row['id'], 'view': row['view'], 'pair': row['pair'], 'gold': row['gold'],
                       'parsed_bound': bound is not None, 'source': selected,
                       'pred': {'direct_scope': raw, 'whole_card': full, 'select_clause': clause},
                       'ms': {'direct_scope': direct_ms, 'whole_card': full_ms, 'select_clause': clause_ms},
                       'quotes': {'whole_card': quote, 'select_clause': clause_quote}})
    return output


def summarize(rows):
    summary = {}
    for method in METHODS:
        views = {}
        for group in ('intact', 'occluded', *CONTROLS):
            cohort = [r for r in rows if r['view'] == group]
            pred = [r['pred'][method] for r in cohort]
            expected = [r['gold'] for r in cohort]
            pair = defaultdict(list)
            for r in cohort:
                pair[r['pair']].append(r['pred'][method] == r['gold'])
            summary_ms = sorted(r['ms'][method] for r in cohort)
            views[group] = {
                'cases': len(cohort), 'accuracy_count': sum(a == b for a, b in zip(pred, expected, strict=True)),
                'wrong_approval': sum(a == 'eligible' and b != 'eligible' for a, b in zip(pred, expected, strict=True)),
                'pairs_correct': sum(all(v) for v in pair.values() if len(v) == 2),
                'p95_ms': summary_ms[int(.95 * (len(summary_ms) - 1))],
            }
        summary[method] = views
    return summary


def main(output):
    import torch

    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires Slurm CUDA')
    cases = challenge()
    if len({r['id'] for r in cases}) != 240 or any(constraint(r['query']) is None for r in cases):
        raise ValueError('Unexpected diagnostic inventory or unsupported bound')
    judge = Judge()
    evaluate(cases[:3], judge)
    rows = evaluate(cases, judge)
    output.mkdir(parents=True, exist_ok=False)
    for name, content in (('cases', cases), ('rows', rows)):
        (output / f'{name}.json').write_text(json.dumps(content, indent=2) + '\n')
    source_reference = {r['id']: r['gold_source'] for r in generate() if r['view'] == 'intact'}
    intact = [(row, case) for row, case in zip(rows, cases, strict=True) if row['view'] == 'intact']
    source_errors = Counter()
    for row, case in intact:
        verified = source_reference[row['id']]
        inferred, _ = evaluate_measurement(constraint(case['query']), case['cards'][verified])
        source_errors['verified_card_correct'] += int(inferred == row['gold'])
        if row['pred']['whole_card'] != row['gold']:
            kind = ('abstained' if row['source'] is None else
                    'wrong_card' if row['source'] != verified else 'correct_card')
            source_errors[kind] += 1
    report = {
        'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
        'source_data': 'Derived from the public 144-case synthetic benchmark and deterministic stress edits.',
        'script_sha256': sha(Path(__file__)), 'model_sha256': judge.sha,
        'files_sha256': {name: sha(output / f'{name}.json') for name in ('cases', 'rows')},
        'views': dict(Counter(r['view'] for r in rows)), 'results': summarize(rows),
        'intact_error_attribution': dict(source_errors),
        'scope': 'Synthetic settings and provided metric and scope. Resolve values and comparators from '
                 'visible raw text; no numeric sidecar. Multi-clause model recheck is charged in full per '
                 'request. Reused original scenarios do not establish industrial transfer.',
    }
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args().output)
