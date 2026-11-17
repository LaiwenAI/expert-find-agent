"""Audit saved independent and setwise evidence-selection costs and coverage."""
import hashlib
import json
import statistics
from pathlib import Path

from expert_search.evidencebench_data import prepare, aspect_coverage
from expert_search.statistics import paired_interval


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output):
    native = Path('artifacts/evidencebench-decisions-v1')
    sets = Path('artifacts/evidencebench-set-selector-v1')
    protocol = json.loads((sets / 'protocol.json').read_text())
    data = prepare()
    native_protocol = json.loads((native / 'protocol.json').read_text())
    if sha(Path('artifacts/public/evidencebench/dev.json')) != native_protocol['dataset_sha256']:
        raise ValueError('Changed EvidenceBench dataset')
    if set(protocol['parents']) != {key + '.json' for key in data}:
        raise ValueError('Changed case inventory')
    rows = []
    for key, source in data.items():
        path = native / (key + '.json')
        if sha(path) != protocol['parents'][path.name]: raise ValueError('Changed native source')
        baseline = json.loads(path.read_text()); candidate = json.loads((sets / path.name).read_text())
        if baseline['id'] != key or candidate['id'] != key or baseline['status'] != 'valid':
            raise ValueError('Invalid or missing paired case')
        if aspect_coverage(source, baseline['selected']) != baseline['metrics']:
            raise ValueError('Changed native coverage')
        if candidate['status'] == 'valid':
            if aspect_coverage(source, candidate['selected']) != candidate['metrics']:
                raise ValueError('Changed set coverage')
            allowed = {int(x[1:]) for x in baseline['decision']['scores']}
            if not set(candidate['selected']) <= allowed: raise ValueError('Foreign sentence')
        elif candidate['metrics'] != baseline['same_query_baseline']:
            raise ValueError('Changed fallback')
        nat_usage = baseline['decision']['response'].get('usage') or {}
        set_usage = candidate.get('response', {}).get('usage') or {}
        def tokens(usage, name):
            value = usage.get(name)
            return value if type(value) is int and value >= 0 else None
        rows.append({'id': key, 'set_valid': candidate['status'] == 'valid',
                     'native': baseline['metrics'], 'set': candidate['metrics'],
                     'native_seconds': baseline['api_wall_seconds'], 'set_seconds': candidate['request_seconds'],
                     'native_input_tokens': tokens(nat_usage, 'input_tokens'),
                     'set_input_tokens': tokens(set_usage, 'prompt_tokens')})
    def aggregate(subset):
        result = {'cases': len(subset), 'set_valid': sum(r['set_valid'] for r in subset)}
        for arm in ('native', 'set'):
            result[arm] = {'macro_aspect_recall': statistics.mean(r[arm]['aspect_recall'] for r in subset),
                           'micro_aspect_recall': sum(r[arm]['covered_aspects'] for r in subset) / sum(r[arm]['total_aspects'] for r in subset),
                           'mean_request_seconds': statistics.mean(r[arm + '_seconds'] for r in subset)}
            known = [r[arm + '_input_tokens'] for r in subset if r[arm + '_input_tokens'] is not None]
            result[arm]['input_tokens_recorded'] = len(known)
            result[arm]['mean_input_tokens_when_recorded'] = statistics.mean(known) if known else None
        result['macro_difference'] = paired_interval([r['set']['aspect_recall'] - r['native']['aspect_recall'] for r in subset])
        return result
    result = {'all_attempts': aggregate(rows), 'valid_set_only': aggregate([r for r in rows if r['set_valid']]),
              'input_sha256': {'native_protocol': sha(native / 'protocol.json'), 'set_protocol': sha(sets / 'protocol.json'),
                               'set_records': {r['id']: sha(sets / (r['id'] + '.json')) for r in rows}},
              'code_sha256': sha(Path(__file__)),
              'scope': 'Post-hoc EvidenceBench development cases, with two failed set calls retaining the semantic baseline as fallback. Single sequential historical API timings are not paired speed estimates; input token counts use distinct API fields, exclude missing usage, and are not price or total system cost. Valid-only is selected on success. Coverage labels do not measure expert fit, attribution or semantic correctness; no SetR checkpoint reproduced.'}
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'input_sha256'}, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
