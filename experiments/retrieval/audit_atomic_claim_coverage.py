"""Check exact-text coverage of provisional recommendation claim extraction."""
import hashlib
import json
import unicodedata
from pathlib import Path

PARENT=Path('artifacts/anchored-temporal-candidate-pilot-v1')
CLAIMS=Path('artifacts/anchored-atomic-claims-audit-v1')


def covered_characters(reason, claims):
    """Count non-whitespace/non-punctuation characters covered by exact spans."""
    if not isinstance(reason,str) or not reason:raise ValueError('Empty reason')
    mask=[False]*len(reason);ambiguous=0
    for claim in claims:
        text=claim['text']
        if not isinstance(text,str) or not text.strip():raise ValueError('Nonliteral claim')
        start=reason.find(text)
        if start<0:raise ValueError('Nonliteral claim')
        if reason.find(text,start+1)>=0:ambiguous+=1
        for i in range(start,start+len(text)):mask[i]=True
    meaningful=[i for i,c in enumerate(reason) if not c.isspace() and not unicodedata.category(c).startswith('P')]
    return dict(meaningful_characters=len(meaningful),covered_characters=sum(mask[i] for i in meaningful),
                ambiguous_claims=ambiguous,claims=len(claims))


def main():
    prior=json.loads((PARENT/'protocol.json').read_text());protocol=json.loads((CLAIMS/'protocol.json').read_text());rows=[]
    for qid in prior['queries']:
        path=PARENT/(qid+'.json');raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=protocol['source_sha256'][qid]:raise ValueError('Changed parent')
        parent=json.loads(raw)
        if parent['status']!='valid':continue
        for i,result in enumerate(parent['response']['results']):
            cid=f'{qid}-{i}';record=json.loads((CLAIMS/'records'/(cid+'.json')).read_text())
            if record['status']=='valid':
                stats=covered_characters(result['reason'],record['claims'])
                stats['source_fact_covered']=covered_characters(result['reason'],[c for c in record['claims'] if c['kind']=='source_fact'])['covered_characters']
                rows.append(stats)
    if len(rows)!=12:raise ValueError('Incomplete cohort')
    total=sum(r['meaningful_characters'] for r in rows);covered=sum(r['covered_characters'] for r in rows)
    print(json.dumps(dict(recommendations=len(rows),meaningful_characters=total,covered_characters=covered,
                          micro_coverage=covered/total,source_fact_character_fraction=sum(r['source_fact_covered'] for r in rows)/total,complete_text_coverage=sum(r['meaningful_characters']==r['covered_characters'] for r in rows),
                          ambiguous_claims=sum(r['ambiguous_claims'] for r in rows),
                          scope='Post-hoc literal character coverage on 12 reasons; punctuation and whitespace omitted. Coverage is not semantic completeness, claim correctness or personal attribution. First occurrence is used for ambiguous repeated substrings.'),indent=2))


if __name__=='__main__':main()
