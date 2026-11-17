"""Measure literal parent-derived text reuse without exporting source text."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def compact(text):
    return re.sub(r'\s+','',text).casefold()


def has_shared_run(source,target,length):
    """Find an exact normalized character run; no fuzzy or semantic matching."""
    if len(target)<length:return False
    return any(target[i:i+length] in source for i in range(len(target)-length+1))


def main():
    root=Path('artifacts/dataset-release/bilingual-query-construction-v1');manifest=json.loads((root/'manifest.json').read_text())
    derived=root/'synthetic-queries.jsonl';parent=Path('artifacts/large-bilingual-data-v1/queries-zh.jsonl')
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    if sha(derived)!=manifest['outputs_sha256'][derived.name] or sha(parent)!=manifest['input_sha256'][str(parent)]:raise ValueError('Changed dataset source')
    parents={r['id']:r for r in map(json.loads,parent.read_text().splitlines())};counts={lang:Counter() for lang in ('en','zh')};total=0
    for row in map(json.loads,derived.read_text().splitlines()):
        total+=1;p=parents[row['parent_id']];source=compact(p['zh'])
        company=compact(p['company']) if isinstance(p.get('company'),str) else ''
        url=compact(p['url']) if isinstance(p.get('url'),str) else ''
        for lang in counts:
            target=compact(row[lang])
            if len(company)>=3 and company in target:counts[lang]['full_company_literal']+=1
            if len(url)>=8 and url in target:counts[lang]['full_url_literal']+=1
            for size in (16,32):
                if has_shared_run(source,target,size):counts[lang]['parent_run_'+str(size)]+=1
    print(json.dumps({'pairs':total,'counts':{k:dict(v) for k,v in counts.items()},'input_sha256':{'derived':sha(derived),'parents':sha(parent)},
                      'scope':'Exact case-folded whitespace-stripped substring checks only. Company fragments, translated identities, nonliteral paraphrases, privacy, licensing and semantic provenance require separate review. A shared passage is expected from grounded synthesis and is not itself a rights determination. No text or identifiers are printed.'},indent=2))


if __name__=='__main__':main()
