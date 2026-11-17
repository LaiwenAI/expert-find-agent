"""Check the local English view without printing text or identifiers."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from .audit_parent_reuse import compact,has_shared_run
from .audit_release_identifiers import PATTERNS


def main():
    root=Path('artifacts/dataset-release/english-repair-construction-v1');manifest=json.loads((root/'manifest.json').read_text());raw=(root/'queries.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest['output_sha256']:raise ValueError('Changed English view')
    parent_path=Path('artifacts/large-bilingual-data-v1/queries-zh.jsonl');original=Path('artifacts/dataset-release/bilingual-query-construction-v1/manifest.json')
    original_manifest=json.loads(original.read_text())
    if hashlib.sha256(parent_path.read_bytes()).hexdigest()!=original_manifest['input_sha256'][str(parent_path)]:raise ValueError('Changed parents')
    parents={r['id']:r for r in map(json.loads,parent_path.read_text().splitlines())};rows=[json.loads(s) for s in raw.splitlines()];counts=Counter()
    if len(rows)!=5000:raise ValueError('Unexpected view count')
    for row in rows:
        if row['english'] is None:counts['missing']+=1;continue
        text=row['english'];parent=parents[row['parent_id']];target=compact(text);source=compact(parent['zh'])
        counts['han_or_radical']+=bool(PATTERNS['han_or_radical'].search(text))
        counts['identifier_rule']+=any(pattern.search(text) for name,pattern in PATTERNS.items() if name!='han_or_radical')
        company=compact(parent['company']) if isinstance(parent.get('company'),str) else ''
        url=compact(parent['url']) if isinstance(parent.get('url'),str) else ''
        counts['full_parent_company']+=len(company)>=3 and company in target
        counts['full_parent_url']+=len(url)>=8 and url in target
        for size in (16,32):counts[f'parent_run_{size}']+=has_shared_run(source,target,size)
    print(json.dumps(dict(rows=len(rows),counts=dict(counts),output_sha256=manifest['output_sha256'],
                          scope='Exact lexical checks only; no claim that remaining text is anonymous, independently verified, rights-cleared, or ready for distribution. Company aliases, translated identities and indirect identifiers may remain.'),indent=2))


if __name__=='__main__':main()
