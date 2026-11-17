"""Export traceable local real/synthetic bilingual data, retaining unresolved slots."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export():
    data = Path('artifacts/large-bilingual-data-v1')
    synthesis = data / 'synthesis-v1'
    protocol = json.loads((synthesis / 'protocol.json').read_text())
    postroot = data / 'postvalidation-v2'
    post = json.loads((postroot / 'manifest.json').read_text())
    if sha(postroot / 'pairs.jsonl') != post['output_sha256']:
        raise ValueError('Changed post-validation records')
    checks = {(r['parent_id'], r['intent_type']): r for r in map(json.loads, (postroot / 'pairs.jsonl').read_text().splitlines())}
    parents = {r['id']: r for r in map(json.loads, (data / 'queries-zh.jsonl').read_text().splitlines())}
    if sha(data / 'queries-zh.jsonl') != protocol['parent_queries_sha256']:
        raise ValueError('Changed real parent cohort')
    real, derived = [], []
    for qid in protocol['selected_parent_ids']:
        parent = parents[qid]
        if parent['split'] != 'construction':
            raise ValueError('Protected split entered synthesis export')
        record = synthesis / 'parents' / (qid + '.json')
        if not record.exists() or sha(record) != post['inputs_sha256'].get(qid):
            raise ValueError('Need complete parent generation and current post-validation snapshot')
        value = json.loads(record.read_text())
        if value['parent_id'] != qid or value['parent_text_sha256'] != parent['source']['text_sha256']:
            raise ValueError('Changed source parent identity')
        real.append(dict(id=qid, group_id=parent['group_id'], split=parent['split'], zh=parent['zh'], en=None, category=parent['category'], track=parent['track'], origin='real_enterprise_need', source=parent['source'], translation_status='complete_parent_translation_pending', expert_fit_grade=None))
        for kind in protocol['taxonomy']:
            variant = value['variants'].get(kind)
            checked = checks.get((qid, kind))
            if variant is not None and checked is None:
                raise ValueError('Missing variant postcheck')
            derived.append(dict(id=qid + ':' + kind, parent_id=qid, group_id=parent['group_id'], split=parent['split'], intent_type=kind, zh=variant['zh'] if variant else None, en=variant['en'] if variant else None, source_ids=variant['source_ids'] if variant else [], origin='model_derived_search_intent', status=checked['status'] if checked else 'unresolved_generation_slot', expert_fit_grade=None))
    if len(real) != 1000 or len(derived) != 5000 or len({r['id'] for r in derived}) != 5000:
        raise ValueError('Incomplete or duplicate planned query slots')
    output = Path('artifacts/dataset-release/bilingual-query-construction-v1')
    output.mkdir(exist_ok=True)
    for name, rows in [('real-queries', real), ('synthetic-queries', derived)]:
        (output / (name + '.jsonl')).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    statuses = {status: sum(r['status'] == status for r in derived) for status in sorted({r['status'] for r in derived})}
    manifest = dict(real_original_queries=len(real), synthetic_slots=len(derived), generated_pairs=sum(r['zh'] is not None and r['en'] is not None for r in derived), statuses=statuses, taxonomy=protocol['taxonomy'], expert_fit_labels=0, complete_original_english_translations=0,
                    input_sha256={str(p): sha(p) for p in [synthesis / 'protocol.json', postroot / 'manifest.json', postroot / 'pairs.jsonl', data / 'queries-zh.jsonl']},
                    outputs_sha256={p.name: sha(p) for p in output.glob('*.jsonl')}, code_sha256=sha(Path(__file__)),
                    scope='Local construction dataset, not public/human-certified relevance benchmark. Retains unresolved slots; screened is same-model plus limited numeric check. One intent has parallel language views, not two independent queries. All selected real parents are construction split; original full Chinese preserved, full English pending. No expert-fit labels inherited or invented. Raw enterprise texts retain their original access terms.')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('input_sha256', 'outputs_sha256', 'taxonomy')}, indent=2))


if __name__ == '__main__':
    export()
