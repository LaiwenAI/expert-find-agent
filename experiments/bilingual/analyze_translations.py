"""Full-cohort accounting for matched original-translation development pilots."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def run():
    data = Path('artifacts/large-bilingual-data-v1')
    roots = dict(direct=data / 'original-translations-v1', placeholders=data / 'original-translations-protected-v1', fixed_blocks=data / 'original-translations-gaps-v1')
    report, records, inputs = {}, {}, {}
    expected_ids = json.loads((roots['direct'] / 'protocol.json').read_text())['selected_parent_ids'][:32]
    for arm, root in roots.items():
        protocol = json.loads((root / 'protocol.json').read_text())
        if protocol['selected_parent_ids'][:32] != expected_ids:
            raise ValueError('Changed matched original cohort')
        rows = []
        for qid in expected_ids:
            path = root / 'parents' / (qid + '.json')
            if not path.exists():
                raise ValueError('Refuse partial translation comparison')
            row = json.loads(path.read_text())
            if row['parent_id'] != qid or row['split'] != 'construction':
                raise ValueError('Changed original/split')
            rows.append(row)
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        records[arm] = {r['parent_id']: r for r in rows}
        report[arm] = dict(terminal_jobs=32, full_translations=sum(r['en'] is not None for r in rows), statuses=dict(Counter(r['status'] for r in rows)), native_audit_labels=dict(Counter(r.get('audit', {}).get('labels', {}).get('full_translation', 'missing') for r in rows)), error_types=dict(Counter(r['error_type'] for r in rows if 'error_type' in r)), numeric_segment_statuses=dict(Counter(s for r in rows for s in r.get('numeric_statuses', []))), mean_job_seconds=float(np.mean([r['seconds'] for r in rows])), protocol_sha256=hashlib.sha256((root / 'protocol.json').read_bytes()).hexdigest())
    contrasts = {}
    rng = np.random.default_rng(20261117)
    for arm in ['placeholders', 'fixed_blocks']:
        differences = np.asarray([int(records[arm][qid]['status'] == 'model_screened_complete_translation') - int(records['direct'][qid]['status'] == 'model_screened_complete_translation') for qid in expected_ids])
        samples = np.mean(differences[rng.integers(0, 32, size=(2000, 32))], axis=1)
        contrasts[arm + '_minus_direct'] = dict(screened_rate_difference=float(differences.mean()), query_bootstrap_interval=np.quantile(samples, [.025, .975]).tolist(), improved_cases=int((differences>0).sum()), worsened_cases=int((differences<0).sum()))
    result = dict(pairs=32, arms=report, paired_contrasts=contrasts, inputs_sha256=inputs, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), scope='Sequential development ablations after observing earlier failures, not randomized concurrent trial or held-out evaluation. Same-model screens and conservative grammar are not human translation quality. Timeouts/structural failures remain in all32 denominator; no retries. Screening-rate contrasts cannot establish correctness improvement. Native-model endpoint behavior and network time may vary between runs.')
    out = data / 'translation-comparison-v1'
    out.mkdir(exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('inputs_sha256', 'code_sha256')}, indent=2))


if __name__ == '__main__':
    run()
