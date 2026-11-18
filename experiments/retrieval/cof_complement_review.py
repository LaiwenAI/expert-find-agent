"""Export and validate blind, token-matched CoF source-complement reviews."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from random import Random

from tokenizers import Tokenizer

from expert_search.cof_benchmark import load
from expert_search.retrieval import tokens

ROOT = Path('artifacts/cof-live-complementary-source-v1')
CHECKPOINT = Path('artifacts/models/qwen3-reranker-0.6b')
LABELS = ('new_support', 'redundant_support', 'conflict', 'insufficient', 'uncertain')
FIELDS = ('option_a', 'option_b')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True)+'\n')


def export(out):
    root, papers, reviewers, _ = load('SIGIR')
    report = json.loads((ROOT/'summary.json').read_text())
    rows_path = ROOT/'rows.json'
    if sha(rows_path) != report['rows_sha256']:
        raise ValueError('Changed source-selection log')
    manifest_path = CHECKPOINT/'checkpoint-provenance.json'
    manifest = json.loads(manifest_path.read_text())
    tokenizer_path = CHECKPOINT/'tokenizer.json'
    if sha(manifest_path) != report['models_sha256'][1] or sha(tokenizer_path) != manifest['files']['tokenizer.json']:
        raise ValueError('Changed tokenizer')
    provenance_path = Path('artifacts/cof-gpu-public-input-v1/input-provenance.json')
    if sha(provenance_path) != report['input_sha256']:
        raise ValueError('Changed input provenance')
    provenance = json.loads(provenance_path.read_text())
    for name in ('SIGIR_papers_test.json','SIGIR_reviewers_test.json'):
        path = root/name
        if sha(path) != provenance['input_sha256'][str(path)]:
            raise ValueError('Changed source text or authorship')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    def text(paper):
        return (papers[paper].get('title','')+'\n'+papers[paper].get('abstract','')).strip()
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False).ids
    def decode(ids):
        return tokenizer.decode(ids, skip_special_tokens=False)
    separator = encode('\n\nNext publication:\n')

    eligible = []
    for row in json.loads(rows_path.read_text()):
        dense = row['arms']['two_matched']
        changed = row['arms']['two_complement']
        if dense['pool'] != changed['pool'] or dense['document_tokens'] != changed['document_tokens']:
            raise ValueError('Changed pool or budgets')
        positions = []
        for i, (standard, selected) in enumerate(zip(dense['used_sources'], changed['used_sources'], strict=True)):
            if standard == selected:
                continue
            if len(standard) != 2 or len(selected) != 2 or standard[0] != selected[0]:
                raise ValueError('Expected only second source to change')
            expert = dense['pool'][i]
            if not set(standard + selected) <= set(reviewers[expert]['papers']):
                raise ValueError('Unexpected source ownership')
            first, second, alternative = [encode(text(p)) for p in (standard[0],standard[1],selected[1])]
            cap = min(256,len(first))
            available = cap - len(separator)
            size = min(len(second), available//2)
            first = first[:available-size]
            if size <= 0 or len(alternative) < size or cap != dense['document_tokens'][i]:
                raise ValueError('Recorded token limits differ')
            second, alternative = second[:size], alternative[:size]
            missing = set(tokens(text(row['query_id']))) - set(tokens(decode(first)))
            if len(missing & set(tokens(decode(alternative)))) <= len(missing & set(tokens(decode(second)))):
                raise ValueError('Could not reconstruct the source substitution')
            positions.append((i, expert, standard, selected, first, second, alternative))
        if positions:
            eligible.append((row['query_id'], positions))
    if sum(len(p) for _, p in eligible) != report['replaced_second']:
        raise ValueError('Changed replacement count')

    rng = Random(20261118)
    picked = rng.sample(eligible, min(40,len(eligible)))
    forms, mapping = [], {}
    for i,(query, positions) in enumerate(picked, start=1):
        slot, expert, standard, selected, first, second, alternative = rng.choice(positions)
        case = f'cof-complement-{i:03d}'
        choices = [('dense_second',standard[1],second),('complement',selected[1],alternative)]
        rng.shuffle(choices)
        form = {'case_id':case, 'target':text(query), 'first_source':decode(first),
                'first_tokens':len(first), 'second_tokens_each':len(second), 'reviewer_id':None}
        for field, (_,_,ids) in zip(FIELDS, choices):
            form[field] = {'text':decode(ids), 'label':None, 'target_capability':None, 'evidence_quote':None,
                           'rationale':None}
        forms.append(form)
        mapping[case] = {'query':query, 'expert':expert, 'slot':slot, 'first':standard[0],
                         'options':{f:{'arm':arm,'paper':pid} for f,(arm,pid,_) in zip(FIELDS,choices)}}
    out.mkdir(parents=True, exist_ok=False)
    (out/'blind-review.jsonl').write_text(''.join(json.dumps(f, ensure_ascii=True)+'\n' for f in forms))
    write(out/'mapping.json', mapping)
    write(out/'manifest.json', {'cases':len(forms), 'eligible_queries':len(eligible),
          'positions_replaced':report['replaced_second'], 'source_job':report['job_id'],
          'rows_sha256':sha(rows_path), 'tokenizer_sha256':sha(tokenizer_path),
          'review_sha256':sha(out/'blind-review.jsonl'), 'mapping_sha256':sha(out/'mapping.json'),
          'scope':'Random sample by query, then one substitution per sampled query. Views reconstructed from the pinned tokenizer and verified source text. Arm and relevance labels are hidden. Abstract content can reveal authorship. Sample covers substitutions only.'})
    (out/'REVIEW.md').write_text(
        'Review independently without opening mapping.json or the ranking logs. Judge each option against '
        'the target and first source. Record the specific technical capability and a verbatim quote from the '
        'option for new_support, redundant_support, or conflict. Use insufficient if the excerpt does not '
        'establish a relevant result, and uncertain when the judgment cannot be made reliably. New support '
        'requires a concrete capability absent from the first source. Shared words alone do not qualify. '
        'Use the rationale field to explain the relationship to the first source. Use null for unfinished '
        'options. A judge should not assume individual contribution from a paper association.\n')


def read_review(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result = {r['case_id']:r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate review case')
    return result


def validate(packet, review):
    manifest = json.loads((packet/'manifest.json').read_text())
    original_path = packet/'blind-review.jsonl'
    if sha(original_path) != manifest['review_sha256'] or sha(packet/'mapping.json') != manifest['mapping_sha256']:
        raise ValueError('Changed packet')
    originals, submitted = read_review(original_path), read_review(review)
    if set(originals) != set(submitted) or len(originals) != manifest['cases']:
        raise ValueError('Changed reviewer cohort')
    counts = Counter()
    for key, row in submitted.items():
        original = originals[key]
        if set(row) != set(original):
            raise ValueError('Changed review fields')
        if any(row[f] != original[f] for f in row if f not in (*FIELDS,'reviewer_id')):
            raise ValueError('Changed target or first source')
        for field in FIELDS:
            option = row[field]
            if set(option) != set(original[field]) or option['text'] != original[field]['text']:
                raise ValueError('Changed evidence option')
            label = option['label']
            if label is None:
                counts['missing'] += 1
                continue
            if label not in LABELS or not isinstance(row['reviewer_id'],str) or not row['reviewer_id'].strip():
                raise ValueError('Invalid label or reviewer ID')
            if not isinstance(option['rationale'],str) or not option['rationale'].strip():
                raise ValueError('Missing rationale')
            if label in LABELS[:3]:
                quote, capability = option['evidence_quote'], option['target_capability']
                if not isinstance(quote,str) or not quote.strip() or quote not in option['text']:
                    raise ValueError('Quote does not occur in displayed evidence')
                if not isinstance(capability,str) or not capability.strip():
                    raise ValueError('Missing target capability')
            counts[label] += 1
    return {'cases':len(submitted), 'counts':dict(counts)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('export').add_argument('packet', type=Path)
    check = commands.add_parser('validate')
    check.add_argument('packet', type=Path)
    check.add_argument('review', type=Path)
    args = parser.parse_args()
    if args.command == 'export':
        export(args.packet)
    else:
        print(json.dumps(validate(args.packet,args.review),indent=2))
