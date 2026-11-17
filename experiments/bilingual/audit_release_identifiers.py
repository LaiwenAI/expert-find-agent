"""Count obvious release risks in generated queries without printing source text."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


PATTERNS={
    'han_or_radical':re.compile(r'[\u2e80-\u2fdf\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]'),
    'url':re.compile(r'(?i)\b(?:https?://|www\.)[^\s<>]+'),
    'email':re.compile(r'(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b'),
    'long_digit_run':re.compile(r'(?<!\d)\d{7,}(?!\d)'),
    'english_organization_suffix':re.compile(r'(?i)\b(?:university|institute|corporation|incorporated|co\.,?\s*ltd\.?|limited liability company)\b'),
    'chinese_organization_suffix':re.compile(r'\u516c\u53f8|\u5927\u5b66|\u7814\u7a76\u9662|\u533b\u9662'),
}


def main():
    root=Path('artifacts/dataset-release/bilingual-query-construction-v1');manifest=json.loads((root/'manifest.json').read_text())
    path=root/'synthetic-queries.jsonl';digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=manifest['outputs_sha256'][path.name]:raise ValueError('Changed dataset')
    rows=[json.loads(line) for line in path.read_text().splitlines()];counts={language:Counter() for language in ('en','zh')};english_han_status=Counter()
    for row in rows:
        if isinstance(row.get('en'),str) and PATTERNS['han_or_radical'].search(row['en']):english_han_status[row['status']]+=1
        for language in counts:
            value=row[language]
            if not isinstance(value,str):counts[language]['missing_text']+=1;continue
            for name,pattern in PATTERNS.items():
                if pattern.search(value):counts[language][name]+=1
            if any(pattern.search(value) for name,pattern in PATTERNS.items() if name!='han_or_radical'):counts[language]['any_rule']+=1
    print(json.dumps({'pairs':len(rows),'counts':{language:dict(counter) for language,counter in counts.items()},'english_han_by_status':dict(english_han_status),'input_sha256':digest,
                      'scope':'Conservative lexical flags, not proof of personal or organizational identity, privacy, copyright compliance or release permission. False positives and false negatives are expected. No source text or detected strings are printed.'},indent=2))


if __name__=='__main__':main()
