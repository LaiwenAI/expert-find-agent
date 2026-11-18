"""Replay document-check budgets using pooled TREC 2006 expert judgments."""
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from expert_search.statistics import paired_interval

FILES = {
    'ent06.qrels.expert': 'b9a0fe026b2fb06273d8f30e6a5cc77a4fef8ed68babe8dc7f3e3a91a192b12d',
    'ent06.qrels.expert-docs': 'a31c82f786475dc17e88dcf2af8cad4528400a9bbbf5710f6ad2724000ac6e29',
}
BUDGETS = (5, 10, 20, 40)
POOL_SIZE = 10
TOP_K = 5
TRIALS = 32
SEED = 20261118
VERIFIERS = {
    'perfect': (1.0, 0.0),
    'sensitivity_90_fpr_1': (0.9, 0.01),
    'sensitivity_90_fpr_5': (0.9, 0.05),
}


def load(directory):
    experts = defaultdict(dict)
    documents = defaultdict(lambda: defaultdict(dict))
    for name, checksum in FILES.items():
        data = (directory / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != checksum:
            raise ValueError(f'Changed qrels: {name}')
        for line in data.decode().splitlines():
            fields = line.split()
            if name == 'ent06.qrels.expert':
                topic, _, expert, grade = fields
                target, key = experts[topic], expert
            else:
                topic, expert, _, doc, grade = fields
                target, key = documents[topic][expert], doc
            if key in target or grade not in {'0', '1', '2'}:
                raise ValueError(f'Invalid judgment in {name}')
            target[key] = int(grade)
    if experts.keys() != documents.keys() or len(experts) != 49:
        raise ValueError('Unexpected topic inventory')
    if any(set(documents[t]) - experts[t].keys() for t in experts):
        raise ValueError('Source judgments have no associated expert judgment')
    return experts, documents


def prior(shortlist_labels, held_out):
    totals = Counter()
    positives = Counter()
    for topic, labels in shortlist_labels.items():
        if topic != held_out:
            totals.update(labels.keys())
            positives.update(e for e, grade in labels.items() if grade == 2)
    prevalence = sum(positives.values()) / sum(totals.values())
    return lambda expert: (positives[expert] + 5 * prevalence) / (totals[expert] + 5)


def metrics(candidates, labels, supported, valid_citations, checked):
    # All arms use the same final ranking rule, preserving order within each group.
    ranked = [e for e in candidates if e in supported] + [e for e in candidates if e not in supported]
    selected = ranked[:TOP_K]
    return {
        'supported_p5': sum(e in valid_citations and labels[e] == 2 for e in selected) / TOP_K,
        'expert_p5': sum(labels[e] == 2 for e in selected) / TOP_K,
        'unsupported_promotion_p5': sum(
            e in supported and (e not in valid_citations or labels[e] != 2) for e in selected
        ) / TOP_K,
        'checked': checked,
    }


def replay(candidates, sequences, labels, method):
    used = Counter()
    supported = set()
    valid_citations = set()
    checked = 0
    order = {e: i for i, e in enumerate(candidates)}
    results = {}
    for budget in BUDGETS:
        while checked < budget:
            available = [
                e for e in candidates if used[e] < len(sequences[e])
                and (method != 'unresolved_first' or e not in supported)
            ]
            if not available:
                break
            if method == 'rank_first':
                expert = available[0]
            else:
                expert = min(available, key=lambda e: (used[e], order[e]))
            true_positive, accepted = sequences[expert][used[expert]]
            used[expert] += 1
            checked += 1
            if accepted:
                supported.add(expert)
                if true_positive:
                    valid_citations.add(expert)
        results[budget] = metrics(candidates, labels, supported, valid_citations, checked)
    return results


def analyze(directory):
    experts, documents = load(directory)
    topics = []
    methods = ('rank_first', 'round_robin', 'unresolved_first')
    for topic in sorted(experts, key=int):
        labels = experts[topic]
        score = prior(experts, topic)
        candidates = sorted(labels, key=lambda e: (-score(e), e))[:POOL_SIZE]
        available = {e: documents[topic].get(e, {}) for e in candidates}
        oracle_count = sum(labels[e] == 2 and 2 in available[e].values() for e in candidates)
        trials = {name: [] for name in VERIFIERS}
        for trial in range(TRIALS):
            sequences = {}
            for expert, choices in available.items():
                ids = sorted(choices)
                random.Random(f'{SEED}:{topic}:{expert}:{trial}').shuffle(ids)
                noise = random.Random(f'{SEED}:noise:{topic}:{expert}:{trial}')
                sequences[expert] = [(choices[i] == 2, noise.random()) for i in ids]
            for name, (sensitivity, false_positive_rate) in VERIFIERS.items():
                observations = {
                    e: [(gold, draw < (sensitivity if gold else false_positive_rate)) for gold, draw in pairs]
                    for e, pairs in sequences.items()
                }
                trials[name].append({m: replay(candidates, observations, labels, m) for m in methods})
        aggregate = {}
        for verifier, entries in trials.items():
            aggregate[verifier] = {
                method: {
                    cap: {name: mean(trial[method][cap][name] for trial in entries)
                          for name in entries[0][method][cap]}
                    for cap in BUDGETS
                }
                for method in methods
            }
        topics.append({
            'topic': topic,
            'pool_checked_pairs': sum(map(len, available.values())),
            'initial_expert_p5': metrics(candidates, labels, set(), set(), 0)['expert_p5'],
            'perfect_verification_supported_p5_upper_bound': min(oracle_count, TOP_K) / TOP_K,
            'arms': aggregate,
        })
    by_verifier = {}
    for verifier in VERIFIERS:
        by_budget = {}
        for cap in BUDGETS:
            arms = {}
            for arm in methods:
                arms[arm] = {
                    name: mean(topic['arms'][verifier][arm][cap][name] for topic in topics)
                    for name in topics[0]['arms'][verifier][arm][cap]
                }
            by_budget[cap] = {
                'arms': arms,
                'unresolved_minus_round_robin_supported_p5': paired_interval([
                    topic['arms'][verifier]['unresolved_first'][cap]['supported_p5']
                    - topic['arms'][verifier]['round_robin'][cap]['supported_p5']
                    for topic in topics
                ]),
            }
        by_verifier[verifier] = by_budget
    return topics, {
        'topics': len(topics),
        'pool_size': POOL_SIZE,
        'top_k': TOP_K,
        'trials_per_topic': TRIALS,
        'order_seed': SEED,
        'input_sha256': FILES,
        'initial_expert_p5': mean(topic['initial_expert_p5'] for topic in topics),
        'unlimited_perfect_ranking_upper_bound': mean(
            topic['perfect_verification_supported_p5_upper_bound'] for topic in topics
        ),
        'verifier_profiles': VERIFIERS,
        'results': by_verifier,
        'scope': (
            'Judged expert and expert-document pools only. Expert priors exclude the current topic; '
            'source order is uniformly shuffled within each expert, shared across methods. '
            'The policy observes either the queried pair label or a shared, independently perturbed '
            'binary version of it. Perturbation rates are hypothetical, not measured model rates. '
            'A supported hit requires a grade-2 expert and an accepted and genuinely grade-2 source. '
            'Budgets count expert-document checks, not unique reads or elapsed time. Topic bootstrap '
            'ignores dependencies from repeated candidates. No corpus inference or clause labels.'
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    topics, summary = analyze(args.input)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = args.output / 'topics.json'
    rows.write_text(json.dumps(topics, indent=2) + '\n')
    summary['topics_sha256'] = hashlib.sha256(rows.read_bytes()).hexdigest()
    summary['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
