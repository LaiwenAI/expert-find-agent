"""Replay promotion and stopping rules with noisy document-level evidence."""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean

from experiments.trec_enterprise.expert_evidence_budget import (
    BUDGETS,
    FILES,
    POOL_SIZE,
    SEED,
    TOP_K,
    TRIALS,
    VERIFIERS,
    load,
    metrics,
    prior,
)
from expert_search.statistics import paired_interval

# Each tuple specifies sensitivity, the false-positive rate, and the fraction
# of candidate/trials for which decisions share one noise draw.
NOISE = {
    **{name: (*rates, 0.0) for name, rates in VERIFIERS.items()},
    'sensitivity_90_fpr_5_correlated': (0.9, 0.05, 0.5),
}
# Stop checking after n accepted documents, promote after m, and optionally
# check one extra document immediately on the first acceptance.
RULES = {
    'round_robin': (None, 1, False),
    'single_stop': (1, 1, False),
    'provisional_continue': (2, 1, False),
    'defer_to_two': (2, 2, False),
    'probe_for_two': (2, 2, True),
}


def observations(topic, trial, documents, sensitivity, false_positive_rate, dependence):
    sequences = {}
    for expert, choices in documents.items():
        ids = sorted(choices)
        random.Random(f'{SEED}:{topic}:{expert}:{trial}').shuffle(ids)
        individual = random.Random(f'{SEED}:noise:{topic}:{expert}:{trial}')
        latent = random.Random(f'{SEED}:shared:{topic}:{expert}:{trial}')
        use_shared = latent.random() < dependence
        shared = latent.random()
        seq = []
        for doc in ids:
            positive = choices[doc] == 2
            draw = shared if use_shared else individual.random()
            threshold = sensitivity if positive else false_positive_rate
            seq.append((positive, draw < threshold))
        sequences[expert] = seq
    return sequences


def replay(candidates, sequences, labels, stop_at, promote_at, probe):
    checked = Counter()
    accepted = Counter()
    valid = Counter()
    supported = set()
    spent = 0
    immediate = None
    order = {e: i for i, e in enumerate(candidates)}
    results = {}
    for budget in BUDGETS:
        while spent < budget:
            eligible = [
                e for e in candidates if checked[e] < len(sequences[e])
                and (stop_at is None or accepted[e] < stop_at)
            ]
            if not eligible:
                break
            expert = immediate if immediate in eligible else min(
                eligible, key=lambda e: (checked[e], order[e])
            )
            immediate = None
            positive, passes = sequences[expert][checked[expert]]
            checked[expert] += 1
            spent += 1
            if not passes:
                continue
            accepted[expert] += 1
            valid[expert] += int(positive)
            if accepted[expert] >= promote_at:
                supported.add(expert)
            if probe and accepted[expert] == 1:
                immediate = expert

        cited = {e for e in supported if valid[e]}
        result = metrics(candidates, labels, supported, cited, spent)
        ranking = sorted(candidates, key=lambda e: (e not in supported, order[e]))[:TOP_K]
        result['two_valid_sources_p5'] = sum(
            e in supported and labels[e] == 2 and valid[e] >= 2 for e in ranking
        ) / TOP_K
        result['single_positive_unpromoted'] = sum(
            accepted[e] == 1 and e not in supported for e in candidates
        )
        results[budget] = result
    return results


def topic_replays(candidates, documents, labels, topic, noise_profile):
    trials = []
    for trial in range(TRIALS):
        sequences = observations(topic, trial, documents, *noise_profile)
        trials.append({
            name: replay(candidates, sequences, labels, *rule) for name, rule in RULES.items()
        })
    return {
        arm: {
            cap: {
                metric: mean(trial[arm][cap][metric] for trial in trials)
                for metric in trials[0][arm][cap]
            }
            for cap in BUDGETS
        }
        for arm in RULES
    }


def summarize(topics):
    summary = {}
    for noise in NOISE:
        summary[noise] = {}
        for budget in BUDGETS:
            per_arm = {
                arm: [t['results'][noise][arm][budget] for t in topics] for arm in RULES
            }
            summary[noise][budget] = {
                'means': {
                    arm: {metric: mean(t[metric] for t in rows) for metric in rows[0]}
                    for arm, rows in per_arm.items()
                },
                'paired_supported_p5_minus_single_stop': {
                    arm: paired_interval([
                        a['supported_p5'] - b['supported_p5']
                        for a, b in zip(rows, per_arm['single_stop'], strict=True)
                    ])
                    for arm, rows in per_arm.items() if arm != 'single_stop'
                },
            }
    return summary


def analyze(directory):
    experts, sources = load(directory)
    topics = []
    for topic in sorted(experts, key=int):
        labels = experts[topic]
        score = prior(experts, topic)
        candidates = sorted(labels, key=lambda e: (-score(e), e))[:POOL_SIZE]
        documents = {e: sources[topic].get(e, {}) for e in candidates}
        positives = [sum(grade == 2 for grade in documents[e].values())
                     for e in candidates if labels[e] == 2]
        topics.append({
            'topic': topic,
            'gold_expert_single_support_count': positives.count(1),
            'gold_expert_any_support_count': sum(x > 0 for x in positives),
            'oracle_one_valid_source_p5': min(sum(x >= 1 for x in positives), TOP_K) / TOP_K,
            'oracle_two_valid_sources_p5': min(sum(x >= 2 for x in positives), TOP_K) / TOP_K,
            'results': {
                name: topic_replays(candidates, documents, labels, topic, profile)
                for name, profile in NOISE.items()
            },
        })
    summary = {
        'topics': len(topics),
        'trials_per_topic': TRIALS,
        'qrels_sha256': FILES,
        'noise_profiles': NOISE,
        'policy_parameters': RULES,
        'gold_expert_single_support_count': sum(t['gold_expert_single_support_count'] for t in topics),
        'gold_expert_any_support_count': sum(t['gold_expert_any_support_count'] for t in topics),
        'oracle_one_valid_source_p5': mean(t['oracle_one_valid_source_p5'] for t in topics),
        'oracle_two_valid_sources_p5': mean(t['oracle_two_valid_sources_p5'] for t in topics),
        'results': summarize(topics),
        'scope': (
            'Pooled judgments only. Same candidate pool, shuffled document order, and verifier '
            'draws per topic and trial. The simulated positivity signal controls search; true '
            'source and expert grades score recommendations only. The shared-noise scenario '
            'uses one latent draw for all documents in a fraction of candidate/trials, preserving '
            'marginal sensitivity and false-positive rates. Distinct document IDs do not imply '
            'independent authors, provenance, or observations. Budgets count checks and do not '
            'measure actual model errors, constraint-level support, or elapsed time.'
        ),
    }
    return summary, topics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    summary, topics = analyze(args.input)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = args.output / 'topics.json'
    rows.write_text(json.dumps(topics, indent=2) + '\n')
    summary['topics_sha256'] = hashlib.sha256(rows.read_bytes()).hexdigest()
    summary['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
