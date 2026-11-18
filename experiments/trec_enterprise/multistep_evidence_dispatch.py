"""Compare immediate and fixed-horizon source-check targets on the same TREC states."""

import argparse
from copy import copy
import hashlib
import json
import os
from pathlib import Path
import random
from statistics import mean

import numpy as np
import torch

from expert_search.statistics import paired_interval
from experiments.trec_enterprise.learned_evidence_dispatch import (
    BUDGETS, EPOCHS, EVAL_TRIALS, FEATURES, FILES, FOLDS, METHODS as FIXED_PROTOCOL,
    NOISE, PENALTY, SEED, TRAIN_TRIALS, cases, fit, load, metrics,
)

HORIZON = 5
LEARNED = ('immediate', 'lookahead', 'lookahead_advantage')
METHODS = FIXED_PROTOCOL[:-1] + LEARNED


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def branch_after(state, expert, horizon):
    branch = copy(state)
    branch.observed = state.observed.copy()
    branch.accepted = state.accepted.copy()
    branch.promoted = state.promoted.copy()
    branch.valid = state.valid.copy()
    branch.check(expert)
    for _ in range(horizon - 1):
        if not (available := branch.available()):
            break
        branch.check(branch.choose(available, 'round_robin'))
    return branch


def targets(state, actions, horizon):
    start = state.utility(state.promoted, state.valid)
    branches = [branch_after(state, action, horizon) for action in actions]
    if len({branch.spent for branch in branches}) != 1:
        raise ValueError('Rollout budgets differ across actions')
    terminal = np.array([branch.utility(branch.promoted, branch.valid)
                         for branch in branches], dtype=np.float32)
    reference = actions.index(state.choose(actions, 'round_robin'))
    return terminal - start, terminal - terminal[reference]


def training(queries, experts, documents):
    x, y = [], {m: [] for m in LEARNED}
    for topic, trial, state in cases(queries, queries, experts, documents, range(TRAIN_TRIALS), NOISE['independent']):
        rng = random.Random(f'{SEED}:{topic}:{trial}')
        for step in range(BUDGETS[-1]):
            if not (actions := state.available()):
                break
            x.extend(state.inputs(actions))
            y['immediate'].extend(state.outcomes(actions))
            total, advantage = targets(state, actions, min(HORIZON, BUDGETS[-1] - step))
            y['lookahead'].extend(total)
            y['lookahead_advantage'].extend(advantage)
            behavior = trial % 4
            explore = behavior == 3 or rng.random() < .2
            chosen = rng.choice(actions) if explore else state.choose(actions, FIXED_PROTOCOL[behavior])
            state.check(chosen)
    return np.asarray(x), {mode: np.asarray(values, dtype=np.float32) for mode, values in y.items()}


def evaluate(queries, train, experts, documents, models):
    results = {}
    for profile, noise in NOISE.items():
        for method in METHODS:
            model = models.get(method)
            policy = 'learned' if model else method
            for topic, _, state in cases(queries, train, experts, documents, range(100, 100 + EVAL_TRIALS), noise):
                for budget in BUDGETS:
                    while state.spent < budget and (actions := state.available()):
                        state.check(state.choose(actions, policy, model))
                    result = metrics(state.candidates, state.labels, state.promoted, state.valid, state.spent)
                    results.setdefault((topic, profile, method, budget), []).append(result)
    return [{'topic': t, 'noise': n, 'method': m, 'budget': b,
             **{k: mean(item[k] for item in values) for k in values[0]}}
            for (t, n, m, b), values in results.items()]


def summarize(rows):
    output = {}
    for profile in NOISE:
        budgets = output[profile] = {}
        for budget in BUDGETS:
            arms = {method: sorted(
                (r for r in rows if r['noise'] == profile and r['budget'] == budget and r['method'] == method),
                key=lambda r: r['topic']) for method in METHODS}
            topics = [row['topic'] for row in arms['round_robin']]
            cost = [row['checked'] for row in arms['round_robin']]
            if len(topics) != 49 or any([r['topic'] for r in arm] != topics or
                                         [r['checked'] for r in arm] != cost for arm in arms.values()):
                raise ValueError('Topic or actual verification counts differ')
            keys = ('supported_p5', 'expert_p5', 'unsupported_promotion_p5', 'checked')
            comparisons = {
                f'{method}_minus_{baseline}': paired_interval([
                    a['supported_p5'] - b['supported_p5']
                    for a, b in zip(arms[method], arms[baseline], strict=True)
                ], repeats=4000)
                for method in LEARNED for baseline in ('round_robin', 'immediate') if method != baseline
            }
            budgets[budget] = {'arms': {m: {k: mean(r[k] for r in a) for k in keys} for m, a in arms.items()},
                               'paired_supported_p5': comparisons}
    return output


def main(directory, output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires a Slurm CUDA allocation')
    experts, docs = load(directory)
    topics = sorted(experts, key=int)
    random.Random(SEED).shuffle(topics)
    output.mkdir(parents=True, exist_ok=False)
    rows, folds = [], []
    for fold in range(FOLDS):
        heldout = topics[fold::FOLDS]
        train = [t for t in topics if t not in heldout]
        x, y = training(train, experts, docs)
        if len({len(labels) for labels in y.values()} | {len(x)}) != 1:
            raise ValueError('Training methods must share the same actions')
        models, provenance = {}, {}
        for mode in LEARNED:
            model, changed, weights = fit(x, y[mode], fold)
            file = output / f'fold-{fold}-{mode}.npz'
            np.savez(file, **weights)
            models[mode] = model
            provenance[mode] = {'parameters_changed': changed, 'checkpoint_sha256': sha(file)}
        rows.extend(evaluate(heldout, train, experts, docs, models))
        folds.append({'fold': fold, 'heldout': heldout, 'train': train, 'training_actions': len(x),
                      'models': provenance})
        print('Fold', fold, 'complete', len(x), flush=True)
    detail = output / 'topics.json'
    detail.write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
    summary = {'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
               'protocol_sha256': sha(Path(__file__)), 'rows_sha256': sha(detail), 'qrels': FILES,
               'features': FEATURES, 'folds': folds, 'horizon': HORIZON, 'penalty': PENALTY,
               'train_trials': TRAIN_TRIALS, 'eval_trials': EVAL_TRIALS, 'epochs': EPOCHS, 'noise': NOISE,
               'results': summarize(rows),
               'scope': 'Cross-fitted on reused TREC topics. Targets use simulated outcomes for every '
                        'valid action within trained folds and round-robin continuation. Evaluation '
                        'executes each policy without rollout privilege or gold labels, then scores '
                        'results against pooled judgments. No lexical retrieval, live verifier or service timing.'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.input, args.output)
