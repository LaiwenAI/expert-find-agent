"""Diagnose numeric-screen behavior on frozen translation pilots."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from expert_search.bilingual.postvalidation import check_pair


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data=Path('artifacts/large-bilingual-data-v1');comparison=data/'translation-comparison-v1/summary.json'
    frozen=json.loads(comparison.read_text());roots={'direct':'original-translations-v1','placeholders':'original-translations-protected-v1','fixed_blocks':'original-translations-gaps-v1'}
    parents=json.loads((data/roots['direct']/'protocol.json').read_text())['selected_parent_ids'][:32]
    output={}
    for arm,name in roots.items():
        root=data/name;cohort=json.loads((root/'protocol.json').read_text())['selected_parent_ids'][:32]
        if cohort!=parents:raise ValueError('Changed cohort')
        segments=Counter();parents_status=Counter();by_native=Counter();mismatch_properties=Counter();negation=Counter()
        for qid in parents:
            path=root/'parents'/(qid+'.json')
            if sha(path)!=frozen['inputs_sha256'][str(path)]:raise ValueError('Changed prior record')
            row=json.loads(path.read_text());parents_status[row['status']]+=1
            native=row.get('audit',{}).get('labels',{}).get('full_translation','missing')
            checks=[]
            for segment in row.get('segments',[]):
                actual=check_pair(segment['zh'],segment['en'])
                if actual!=segment['numeric_check']:raise ValueError('Changed numeric check')
                checks.append(actual['status']);segments[actual['status']]+=1
                a,b=actual['source_anchors'],actual['target_anchors']
                if actual['status']=='uncertain':
                    negation['source']+=any(x['uncertain_negation'] for x in a)
                    negation['target']+=any(x['uncertain_negation'] for x in b)
                if actual['status']=='mismatch':
                    av=[x['value'] for x in a];bv=[x['value'] for x in b]
                    if len(av)!=len(bv):mismatch_properties['anchor_count_differs']+=1
                    if Counter(av)!=Counter(bv):mismatch_properties['numeric_multiset_differs']+=1
                    elif av!=bv:mismatch_properties['numeric_order_differs']+=1
                    if av==bv and [x['unit'] for x in a]!=[x['unit'] for x in b]:mismatch_properties['unit_sequence_differs']+=1
                    if av==bv and [x['operator'] for x in a]!=[x['operator'] for x in b]:mismatch_properties['operator_sequence_differs']+=1
            if checks:
                category=('mismatch_and_uncertain' if 'uncertain' in checks and 'mismatch' in checks else
                          'uncertain' if 'uncertain' in checks else 'mismatch' if 'mismatch' in checks else 'match')
                by_native[native,category]+=1
        output[arm]={'parent_statuses':dict(parents_status),'numeric_segment_statuses':dict(segments),
                     'native_by_numeric':[{'native':a,'numeric':b,'parents':v} for (a,b),v in sorted(by_native.items())],
                     'mismatch_properties':dict(mismatch_properties),'uncertain_segment_negation_flags':dict(negation)}
    report={'parents_per_arm':len(parents),'arms':output,'comparison_sha256':sha(comparison),'code_sha256':sha(Path(__file__)),
            'scope':'Read-only post-hoc screen accounting on three sequential 32-parent development pilots. Mismatch properties can overlap; numeric grammar is incomplete and neither its status nor same-model native review is human translation truth. No source or translation text printed.'}
    out=data/'translation-guard-audit-v2';out.mkdir(exist_ok=False);(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
