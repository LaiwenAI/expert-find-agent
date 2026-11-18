"""Fit a Qwen evidence-action adapter using public HotpotQA supporting facts."""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from statistics import quantiles
from time import perf_counter

from experiments.decision.hotpot_evidence_action import (
    DATA_SHA256, HF_REVISION, TRAIN_SHA256, action_features, candidate_actions, examples, normalize, sha, training_rows,
)

SEED = 20261119
SIZES = {'train': 384, 'validation': 128, 'dev': 512}
EPOCHS = 2
BATCH = 8
RANK = 8
PROMPT = (
    'Select one next passage to supply evidence missing from the first passage. '
    'Use the question and the provided excerpts. Output only the number of the next passage.'
)
REVISION = 'cdbee75f17c01a7cc42f958dc650907174af0554'


def facts(row):
    return {normalize(title) for title, _ in row['supporting_facts']}


def keyed(key):
    return hashlib.sha256(f'{SEED}:{key}'.encode()).digest()


def case(row):
    usable = examples(row)
    if usable is None:
        return None
    qid, _, outcomes, lengths, count, _ = usable
    titles, texts, first, actions, ranks = candidate_actions(row['question'], row['context'])
    features = action_features(row['question'], titles, texts, first, actions, ranks)
    return {
        'id': qid, 'question': row['question'], 'first': texts[first],
        'actions': [texts[i] for i in actions], 'outcomes': outcomes[:count].astype(int).tolist(),
        'chars': lengths[:count].tolist(), 'features': features.tolist(),
    }


def prepare(dev, shard, output):
    if sha(dev) != DATA_SHA256 or sha(shard) != TRAIN_SHA256[shard.name]:
        raise ValueError('Changed public data')
    development = json.loads(dev.read_text())
    used = {normalize(row['question']) for row in development}
    dev_ids = {row['_id'] for row in development}
    dev_facts = set().union(*(facts(row) for row in development))
    folds = {k: [] for k in SIZES}
    folds['dev'] = [c for row in sorted(development, key=lambda r: keyed(r['_id']))
                    if (c := case(row)) is not None][:SIZES['dev']]

    for fold in ('validation', 'train'):
        added_facts = set()
        for row in training_rows([shard]):
            row['context'] = list(row['context'])
            row['supporting_facts'] = list(row['supporting_facts'])
            q = normalize(row['question'])
            validation = keyed(row['_id'])[0] % 5 == 0
            if validation != (fold == 'validation') or q in used or row['_id'] in dev_ids:
                continue
            supporting = facts(row)
            if supporting & dev_facts:
                continue
            item = case(row)
            if item is None or len(item['actions']) < 2 or not any(item['outcomes']):
                continue
            folds[fold].append(item)
            used.add(q)
            added_facts.update(supporting)
            if len(folds[fold]) == SIZES[fold]:
                break
        dev_facts.update(added_facts)
    if {key: len(items) for key, items in folds.items()} != SIZES:
        raise ValueError('Unexpected cohort size')
    output.write_text(json.dumps({'folds': folds, 'revision': HF_REVISION,
                                 'raw_sha256': {p.name: sha(p) for p in (dev, shard)}},
                                 ensure_ascii=False, allow_nan=False) + '\n')


def run(pack, head_file, model_root, output, expected_pack_sha):
    import numpy as np
    import torch
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file, save_file

    from expert_search.statistics import paired_interval
    from experiments.decision.hotpot_evidence_action_timing import EvidenceHead

    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires a Slurm GPU allocation')
    if sha(pack) != expected_pack_sha:
        raise ValueError('Changed action cohort')
    provenance = model_root / 'local-provenance.json'
    spec = json.loads(provenance.read_text())
    if spec['revision'] != REVISION or any(sha(model_root / path) != digest for path, digest in spec['files'].items()):
        raise ValueError('Changed model checkpoint')
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision('highest')
    rng = random.Random(SEED)
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_root, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa'
    ).to('cuda').eval().requires_grad_(False)
    digit_ids = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(1, 5)]
    if any(len(ids) != 1 for ids in digit_ids):
        raise ValueError('Expected one token per action')
    label_tokens = torch.tensor([ids[0] for ids in digit_ids], device='cuda')

    class LowRankLinear(nn.Module):
        def __init__(self, linear):
            super().__init__()
            self.linear = linear
            self.down = nn.Linear(linear.in_features, RANK, bias=False, device='cuda')
            self.up = nn.Linear(RANK, linear.out_features, bias=False, device='cuda')
            nn.init.zeros_(self.up.weight)
            self.active = False

        def forward(self, x):
            value = self.linear(x)
            if self.active:
                value = value + (self.up(self.down(x.float())) * 2).to(value.dtype)
            return value

    adapters = []
    for block in model.model.layers[-2:]:
        for name in ('q_proj', 'v_proj'):
            adapter = LowRankLinear(getattr(block.self_attn, name))
            setattr(block.self_attn, name, adapter)
            adapters.append(adapter)
    parameters = {f'{i}.{name}.weight': getattr(adapter, name).weight
                  for i, adapter in enumerate(adapters) for name in ('down', 'up')}
    initial = {k: v.detach().cpu().clone() for k, v in parameters.items()}

    def excerpt(text, limit):
        ids = tokenizer.encode(text, add_special_tokens=False)
        return tokenizer.decode(ids[:limit]) if len(ids) > limit else text

    def prompt(row):
        question = excerpt(row['question'], 64)
        first = excerpt(row['first'], 64)
        choices = '\n'.join(f'{i}. {excerpt(text, 56)}' for i, text in enumerate(row['actions'], 1))
        messages = [{'role': 'system', 'content': PROMPT},
                    {'role': 'user', 'content': f'Question: {question}\nFirst passage: {first}\nOptions:\n{choices}'}]
        return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                             return_dict=False)

    payload = json.loads(pack.read_text())
    splits = {k: [{**row, 'tokens': prompt(row)} for row in rows]
              for k, rows in payload['folds'].items()}
    print('Prompt lengths', {k: max(len(r['tokens']) for r in v) for k, v in splits.items()}, flush=True)

    def score(rows):
        width = max(len(r['tokens']) for r in rows)
        masks = [[0] * (width - len(r['tokens'])) + [1] * len(r['tokens']) for r in rows]
        ids = [[tokenizer.pad_token_id] * (width - len(r['tokens'])) + r['tokens'] for r in rows]
        mask = torch.tensor(masks, device='cuda')
        positions = (mask.cumsum(dim=-1) - 1).clamp_min(0)
        logits = model(input_ids=torch.tensor(ids, device='cuda'), attention_mask=mask, position_ids=positions,
                       use_cache=False, logits_to_keep=1).logits[:, -1, label_tokens].float()
        allowed = torch.tensor([len(r['actions']) for r in rows], device='cuda')
        return logits.masked_fill(torch.arange(4, device='cuda') >= allowed[:, None], -torch.inf)

    def active(enabled):
        for adapter in adapters:
            adapter.active = enabled

    @torch.inference_mode()
    def predict(rows, enabled):
        active(enabled)
        output = []
        for start in range(0, len(rows), BATCH):
            output.extend(score(rows[start:start + BATCH]).argmax(dim=1).cpu().tolist())
        return output

    def complete(rows, choices):
        return sum(row['outcomes'][choice] for row, choice in zip(rows, choices, strict=True))

    val, test = splits['validation'], splits['dev']
    validation_frozen = complete(val, predict(val, False))
    optimizer = torch.optim.AdamW(parameters.values(), lr=2e-4)
    best, history = None, []
    started = perf_counter()
    for epoch in range(1, EPOCHS + 1):
        active(True)
        cases = sorted(splits['train'], key=lambda row: len(row['tokens']))
        batches = [cases[i:i+BATCH] for i in range(0, len(cases), BATCH)]
        rng.shuffle(batches)
        losses = []
        for batch in batches:
            logits = score(batch)
            positives = torch.tensor([r['outcomes'] + [0] * (4 - len(r['actions'])) for r in batch],
                                     dtype=torch.bool, device='cuda')
            loss = (logits.logsumexp(dim=1) - logits.masked_fill(~positives, -torch.inf).logsumexp(dim=1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters.values(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        hits = complete(val, predict(val, True))
        history.append({'epoch': epoch, 'mean_loss': sum(losses) / len(losses), 'complete': hits})
        print(history[-1], flush=True)
        if best is None or hits > best['complete']:
            best = {'epoch': epoch, 'complete': hits, 'state': {k: v.detach().cpu().clone() for k, v in parameters.items()}}
    train_seconds = perf_counter() - started
    adapter_file = output / 'adapter.safetensors'
    save_file(best['state'], adapter_file)
    saved = load_file(adapter_file)
    with torch.no_grad():
        for key, param in parameters.items():
            param.copy_(saved[key])
    changed = sum(int(torch.count_nonzero(saved[k] != v)) for k, v in initial.items())
    if not changed:
        raise ValueError('Adapter parameters did not change')

    head = EvidenceHead(head_file)
    controls = {
        'bm25': [0] * len(test),
        'mlp': [int(head.scores(np.array(r['features'], dtype=np.float32)).argmax()) for r in test],
        'frozen': predict(test, False), 'lora': predict(test, True),
    }
    rows = []
    for i, row in enumerate(test):
        chosen = {k: values[i] for k, values in controls.items()}
        if any(row['chars'][j] > row['chars'][0] for j in chosen.values()):
            raise ValueError('Evidence budget violation')
        rows.append({'id': row['id'], 'tokens': len(row['tokens']), 'oracle': max(row['outcomes']),
                     'choices': chosen, 'had_choice': len(row['actions']) > 1,
                     'complete': {k: row['outcomes'][j] for k, j in chosen.items()},
                     'chars': {k: row['chars'][j] for k, j in chosen.items()}})
    timings = {k: [] for k in ('frozen', 'lora')}
    with torch.inference_mode():
        for i, row in enumerate([r for r in test if len(r['actions']) > 1][:128]):
            arms = [('frozen', False), ('lora', True)]
            for name, enabled in arms[i % 2:] + arms[:i % 2]:
                active(enabled)
                start = perf_counter()
                scored = score([row]).argmax(dim=1).item()
                timings[name].append((perf_counter() - start) * 1000)
                rows_index = controls[name][test.index(row)]
                if scored != rows_index:
                    timings.setdefault(f'{name}_batch_disagreements', []).append(row['id'])
    row_file = output / 'rows.json'
    row_file.write_text(json.dumps(rows, indent=2) + '\n')
    summary = {
        'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
        'code_sha256': sha(Path(__file__)), 'checkpoint_sha256': sha(provenance),
        'pack_sha256': sha(pack), 'mlp_sha256': sha(head_file), 'adapter_sha256': sha(adapter_file),
        'rows_sha256': sha(row_file), 'trainable': sum(v.numel() for v in parameters.values()), 'changed': changed,
        'training_validation_seconds': train_seconds, 'history': history, 'selected_epoch': best['epoch'],
        'validation_frozen_complete': validation_frozen, 'counts': SIZES,
        'dev_multiple_choices': sum(r['had_choice'] for r in rows), 'dev_oracle': sum(r['oracle'] for r in rows),
        'outcomes': {k: {'complete': complete(test, v), 'mean_chars': float(np.mean([r['chars'][k] for r in rows]))}
                     for k, v in controls.items()},
        'paired_lora_minus': {k: paired_interval([r['complete']['lora'] - r['complete'][k] for r in rows], repeats=4000)
                              for k in ('frozen', 'mlp')},
        'batch_disagreements': {k: len(v) for k, v in timings.items() if k.endswith('disagreements')},
        'timed_pairs': len(timings['frozen']),
        'gpu_decision_p95_ms': {k: quantiles(timings[k], n=20, method='inclusive')[-1] for k in ('frozen', 'lora')},
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for arg in ('dev', 'shard', 'output'):
        prep.add_argument(arg, type=Path)
    fit = sub.add_parser('fit')
    for arg in ('pack', 'head', 'model', 'output'):
        fit.add_argument(arg, type=Path)
    fit.add_argument('expected_sha')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.dev, args.shard, args.output)
    else:
        run(args.pack, args.head, args.model, args.output, args.expected_sha)
