"""Compare published expert-only and evidence-supported scores of TREC 2006 runs."""

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
from statistics import median
import subprocess

REVISION = '75ec9339eac60126b1045e5858a5ac874ea54a51'
INDEX_SHA256 = 'ea9587ba14f562aa26e6d481a08485716b431e833fd3393d52f8f3ee1655ee92'
BASE = 'https://trec.nist.gov/pubs/trec15/appendices/enterprise/'
INDEX_URL = (f'https://raw.githubusercontent.com/usnistgov/trec-browser/{REVISION}'
             '/browser/src/docs/trec15/enterprise/results.md')


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique(pattern, text, field):
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f'Expected one {field}; found {len(matches)}')
    return matches[0]


def catalog(index):
    if checksum(index) != INDEX_SHA256:
        raise ValueError(f'Expected pinned official catalog from {INDEX_URL}')
    sections = re.split(r'^#### ([\w-]+)\s*\n', index.read_text(), flags=re.MULTILINE)
    for run, text in zip(sections[1::2], sections[2::2], strict=True):
        url = BASE + run + '.expert.pdf'
        if f']({url})' not in text:
            raise ValueError(f'No matching expert appendix for {run}')
        participant = unique(r'\]\(\./participants\.md#([^)]+)\)', text, 'participant')
        values = {
            metric: float(unique(rf'^[ \t]*{field}[ \t]+all[ \t]+(\d+\.\d+)[ \t]*$', text, field))
            for metric, field in (('map', 'map'), ('p10', 'P_10'))
        }
        yield run, url, participant, values


def appendix(path, run):
    text = subprocess.run(['pdftotext', '-layout', str(path), '-'],
                          check=True, capture_output=True, text=True).stdout
    header = re.search(r'^Enterprise Track results, expert search task[^\n]+\n\s+'
                       + re.escape(run) + r'[ \t]*$', text, re.MULTILINE)
    if not header:
        raise ValueError(f'Wrong run: {path}')
    single = bool(re.search(r'Relevant \(expert and supported\)[ \t]+1393\b', text))
    double = bool(re.search(r'Expert[ \t]+Supported', text)
                  and re.search(r'^\s*Relevant[ \t]+1393[ \t]+1393[ \t]*$', text, re.MULTILINE))
    if single == double:
        raise ValueError(f'Unexpected relevance definition: {path}')
    if int(unique(r'^\s*Number of Topics[ \t]+(\d+)[ \t]*$', text, 'topics')) != 49:
        raise ValueError(f'Unexpected topic inventory: {path}')
    run_type = unique(r'^\s*Run Type[ \t]+(\w+)[ \t]*$', text, 'run type')
    expert, supported = {}, {}
    for metric, field in (('map', 'Average Precision'), ('p5', 'At 5 docs'), ('p10', 'At 10 docs')):
        pattern = rf'{field}[ \t]+(\d+\.\d+)[ \t]+(\d+\.\d+)[ \t]*$' if double else (
            rf'{field}[ \t]+(\d+\.\d+)[ \t]*$')
        values = unique(pattern, text, field)
        if double:
            expert[metric], supported[metric] = map(float, values)
        else:
            supported[metric] = float(values)
    return run_type, expert, supported


def relation(rows, metric):
    counts = dict(concordant=0, discordant=0, expert_ties=0, support_ties=0, joint_ties=0)
    for a, b in combinations(rows, 2):
        expert = a['expert'][metric] - b['expert'][metric]
        supported = a['supported'][metric] - b['supported'][metric]
        if expert and supported:
            counts['concordant' if expert * supported > 0 else 'discordant'] += 1
        elif expert or supported:
            counts['support_ties' if expert else 'expert_ties'] += 1
        else:
            counts['joint_ties'] += 1
    comparable = counts['concordant'] + counts['discordant']
    denominator = ((comparable + counts['expert_ties']) * (comparable + counts['support_ties'])) ** .5
    return {
        **counts, 'kendall_tau_b': (counts['concordant'] - counts['discordant']) / denominator,
        'discordance_among_untied': counts['discordant'] / comparable,
    }


def describe(rows):
    return {
        'runs': len(rows),
        'groups': len({r['participant'] for r in rows}),
        'map_association': relation(rows, 'map'),
        'p10_association': relation(rows, 'p10'),
        'median_p10_gap': median(r['expert']['p10'] - r['supported']['p10'] for r in rows),
        'median_supported_p5': median(r['supported']['p5'] for r in rows),
        'leaders': {mode: max(rows, key=lambda r: r[mode]['p10'])['run'] for mode in ('expert', 'supported')},
    }


def evaluate(index, pdfs):
    records = []
    for run, url, participant, expert in catalog(index):
        path = pdfs / Path(url).name
        kind, check_expert, supported = appendix(path, run)
        if any(check_expert[m] != expert[m] for m in check_expert.keys() & expert.keys()):
            raise ValueError(f'Appendix expert scores disagree with the index: {run}')
        expert.update(check_expert)
        if supported['p10'] > expert['p10']:
            raise ValueError(f'Supported precision exceeds expert precision: {run}')
        records.append({'run': run, 'participant': participant, 'type': kind, 'expert': expert,
                        'supported': supported, 'appendix': url, 'appendix_sha256': checksum(path)})
    if len({r['run'] for r in records}) != len(records) or len(records) != 91:
        raise ValueError('Unexpected official run inventory')
    return {
        'index': INDEX_URL, 'index_sha256': checksum(index), 'topics': 49,
        'all_runs': describe(records),
        'automatic_runs': describe([r for r in records if r['type'] == 'automatic']),
        'positive_supported_runs': describe([r for r in records if r['supported']['map'] > 0]),
        'runs': records,
        'scope': 'Archived aggregate scores, not rescored raw submissions. Paired by run on the same '
                 '49 pooled topics. Missing file or field fails rather than changing the cohort. '
                 'Reversals are descriptive across dependent submissions, not confidence intervals, '
                 'document-level support audits, latency estimates or a current system leaderboard.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index', type=Path)
    parser.add_argument('appendices', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(evaluate(args.index, args.appendices), indent=2, allow_nan=False) + '\n')
