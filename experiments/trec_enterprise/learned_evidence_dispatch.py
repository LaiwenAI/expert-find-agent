"""Train and cross-fit a source-check decision policy on TREC expert judgments."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
from statistics import mean

import numpy as np
import torch

from expert_search.statistics import paired_interval
from experiments.trec_enterprise.expert_evidence_budget import BUDGETS, FILES, POOL_SIZE, TOP_K, load, metrics, prior
from experiments.trec_enterprise.expert_evidence_confirmation import observations

SEED = 20261120
FOLDS = 5
TRAIN_TRIALS = 8
EVAL_TRIALS = 32
EPOCHS = 10
PENALTY = .25
METHODS = ('rank_first', 'round_robin', 'unresolved_first', 'learned')
NOISE = {'independent': (.9, .05, 0.), 'shared_noise': (.9, .05, .5)}
FEATURES = ('initial_rank', 'prior', 'spent', 'checked', 'accepted', 'remaining',
            'number_promoted', 'current_top5', 'top5_after_accept', 'promoted_rank',
            'untouched', 'remaining_share')


class State:
    def __init__(self, candidates, scores, sequences, labels):
        self.candidates, self.scores, self.sequences, self.labels = candidates, scores, sequences, labels
        self.order = {e: i for i, e in enumerate(candidates)}
        self.observed, self.accepted = Counter(), Counter()
        self.promoted, self.valid = set(), set()
        self.spent = 0

    def available(self):
        return [e for e in self.candidates if self.observed[e] < len(self.sequences[e])]

    def rank(self, accepted):
        return sorted(self.candidates, key=lambda e: (e not in accepted, self.order[e]))

    def inputs(self, actions):
        ranked = self.rank(self.promoted)
        values = []
        for e in actions:
            checked = self.observed[e]
            size = len(self.sequences[e])
            if e in self.promoted:
                next_rank = ranked.index(e)
            else:
                next_rank = sum(self.order[p] < self.order[e] for p in self.promoted)
            values.append([self.order[e], self.scores[e], self.spent, checked, self.accepted[e],
                           size - checked, len(self.promoted), ranked.index(e) < TOP_K,
                           next_rank < TOP_K, next_rank, checked == 0, (size - checked) / size])
        return np.array(values, dtype=np.float32)

    def outcomes(self, actions):
        reward = self.utility(self.promoted, self.valid)
        gains = []
        for e in actions:
            gold, passed = self.sequences[e][self.observed[e]]
            if not passed:
                gains.append(0.)
                continue
            valid = self.valid | {e} if gold else self.valid
            gains.append(self.utility(self.promoted | {e}, valid) - reward)
        return gains

    def utility(self, promoted, valid):
        measures = metrics(self.candidates, self.labels, promoted, valid, self.spent)
        return measures['supported_p5'] - PENALTY * measures['unsupported_promotion_p5']

    def check(self, expert):
        gold, passed = self.sequences[expert][self.observed[expert]]
        self.observed[expert] += 1
        self.spent += 1
        if passed:
            self.promoted.add(expert)
            self.accepted[expert] += 1
            if gold:
                self.valid.add(expert)

    def choose(self, actions, method, model=None):
        if method == 'learned':
            return actions[int(model(self.inputs(actions)).argmax())]
        if method == 'rank_first':
            return actions[0]
        if method == 'unresolved_first':
            return min(actions, key=lambda e: (e in self.promoted, self.observed[e], self.order[e]))
        return min(actions, key=lambda e: (self.observed[e], self.order[e]))


def cases(queries, prior_queries, experts, documents, trials, noise):
    base = {t: experts[t] for t in prior_queries}
    for topic in queries:
        score = prior(base, topic)
        known = experts[topic]
        candidates = sorted(known, key=lambda e: (-score(e), e))[:POOL_SIZE]
        scores = {e: score(e) for e in candidates}
        source_pool = {e: documents[topic].get(e, {}) for e in candidates}
        for trial in trials:
            sequence = observations(topic, trial, source_pool, *noise)
            yield topic, trial, State(candidates, scores, sequence, known)


def build_training(queries, experts, documents):
    x, y = [], []
    for topic, trial, state in cases(queries, queries, experts, documents, range(TRAIN_TRIALS), NOISE['independent']):
        rng = random.Random(f'{SEED}:{topic}:{trial}')
        for step in range(BUDGETS[-1]):
            actions = state.available()
            if not actions:
                break
            x.extend(state.inputs(actions))
            y.extend(state.outcomes(actions))
            mode = trial % 4
            exploration = mode == 3 or rng.random() < .2
            picked = rng.choice(actions) if exploration else state.choose(actions, METHODS[mode])
            state.check(picked)
    return np.asarray(x), np.asarray(y, dtype=np.float32)


def fit(x, y, fold):
    torch.manual_seed(SEED + fold)
    tx, ty = (torch.from_numpy(v).to('cuda') for v in (x, y))
    center = tx.mean(0)
    scale = tx.std(0).clamp_min(1e-6)
    tx = (tx - center) / scale
    net = torch.nn.Sequential(torch.nn.Linear(len(FEATURES), 16), torch.nn.ReLU(),
                              torch.nn.Linear(16, 1)).to('cuda')
    start = [p.detach().clone() for p in net.parameters()]
    optimizer = torch.optim.AdamW(net.parameters(), lr=.003)
    for _ in range(EPOCHS):
        for idx in torch.randperm(len(tx), device='cuda').split(256):
            prediction = net(tx[idx]).squeeze(1)
            loss = torch.nn.functional.mse_loss(prediction, ty[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    changed = sum(int(torch.count_nonzero(a != b)) for a, b in zip(start, net.parameters(), strict=True))
    weight = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
    center, scale = center.cpu().numpy(), scale.cpu().numpy()

    def predict(features):
        hidden = np.maximum((features - center) / scale @ weight['0.weight'].T + weight['0.bias'], 0)
        return (hidden @ weight['2.weight'].T + weight['2.bias']).ravel()

    with torch.inference_mode():
        ref = net(tx[:256]).squeeze(1).cpu().numpy()
    np.testing.assert_allclose(predict(x[:256]), ref, atol=1e-6, rtol=1e-5)
    return predict, changed, {'center': center, 'scale': scale, **weight}


def evaluate(topics, train, experts, documents, model):
    rows = {}
    for profile, noise in NOISE.items():
        for method in METHODS:
            for topic, _, state in cases(topics, train, experts, documents, range(100, 100 + EVAL_TRIALS), noise):
                for cap in BUDGETS:
                    while state.spent < cap and (actions := state.available()):
                        state.check(state.choose(actions, method, model))
                    row = metrics(state.candidates, state.labels, state.promoted, state.valid, state.spent)
                    rows.setdefault((topic, profile, method, cap), []).append(row)
    return [{'topic': t, 'noise': n, 'method': m, 'budget': b, **{
        k: mean(item[k] for item in values) for k in values[0]
    }} for (t, n, m, b), values in rows.items()]


def summarize(rows):
    summary = {}
    for profile in NOISE:
        summary[profile] = {}
        for cap in BUDGETS:
            matched = {m: sorted((r for r in rows if r['noise'] == profile and r['budget'] == cap
                                  and r['method'] == m), key=lambda r: r['topic']) for m in METHODS}
            assert all(len(r) == 49 for r in matched.values())
            assert all({r[i]['checked'] for r in matched.values()} == {matched['learned'][i]['checked']}
                       for i in range(49))
            keys = ('supported_p5', 'expert_p5', 'unsupported_promotion_p5', 'checked')
            summary[profile][cap] = {
                'means': {m: {k: mean(r[k] for r in values) for k in keys} for m, values in matched.items()},
                'paired': {m: paired_interval([
                    a['supported_p5'] - b['supported_p5'] for a, b in zip(matched['learned'], values, strict=True)
                ], repeats=4000) for m, values in matched.items() if m != 'learned'},
            }
    return summary


def main(directory, output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Needs a Slurm CUDA allocation')
    experts, documents = load(directory)
    topics = sorted(experts, key=int)
    random.Random(SEED).shuffle(topics)
    output.mkdir(parents=True, exist_ok=False)
    rows, folds = [], []
    for fold in range(FOLDS):
        heldout = topics[fold::FOLDS]
        train = [t for t in topics if t not in heldout]
        x, y = build_training(train, experts, documents)
        model, changed, weights = fit(x, y, fold)
        checkpoint = output / f'fold-{fold}.npz'
        np.savez(checkpoint, **weights)
        folds.append({'heldout': heldout, 'train': train, 'training_actions': len(y),
                      'changed_parameters': changed, 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
        rows.extend(evaluate(heldout, train, experts, documents, model))
        print('Fold', fold, 'complete', len(y), flush=True)
    path = output / 'topics.json'
    path.write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
    result = {'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(), 'qrels': FILES,
              'rows_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'folds': folds, 'train_trials': TRAIN_TRIALS, 'evaluation_trials': EVAL_TRIALS, 'epochs': EPOCHS,
              'noise': NOISE, 'features': FEATURES, 'judgment_penalty': PENALTY, 'results': summarize(rows),
              'scope': 'Previously examined TREC topics cross-fitted by topic. Shared shortlist uses only '
                       'currently judged candidate IDs with an out-of-fold expert prior. Verification is '
                       'simulated from judged documents, and budget counts calls. No open-corpus retrieval, '
                       'real verifier, independent expert cohort or end-to-end latency.'}
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.input, args.output)
