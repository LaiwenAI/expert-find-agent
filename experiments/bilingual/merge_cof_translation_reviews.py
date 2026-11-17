"""Merge disjoint blind CoF translation packets without exposing screen status."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from .check_cof_translation_review import check,read


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(packets,output):
    cases=[];seen=set();missing=0;hashes={}
    for packet in packets:
        baseline=packet/'blind-review.jsonl';mapping_path=packet/'private-mapping.json';manifest=json.loads((packet/'manifest.json').read_text())
        check(packet,baseline);forms=read(baseline);mapping=json.loads(mapping_path.read_text())
        if len(mapping)!=len(forms) or len({r['case_id'] for r in mapping})!=len(mapping) or {r['case_id'] for r in mapping}!=set(forms):raise ValueError('Changed packet mapping')
        for item in mapping:
            row=forms[item['case_id']]
            if item['query_id'] in seen or any(row[k] is not None for k in ('equivalence','rationale','reviewer_id')):raise ValueError('Duplicate query or nonblank baseline')
            seen.add(item['query_id']);cases.append((row,item))
        missing+=manifest['missing_translation_cases'];hashes[str(packet/'manifest.json')]=sha(packet/'manifest.json')
    if len(cases)!=72 or missing!=1:raise ValueError('Changed planned 73-query review coverage')
    random.Random(20261124).shuffle(cases);forms=[];mapping=[]
    for i,(row,item) in enumerate(cases,1):
        cid=f'cof-{i:03d}'
        forms.append(dict(case_id=cid,english_source=row['english_source'],chinese_translation=row['chinese_translation'],equivalence=None,rationale=None,reviewer_id=None))
        mapping.append(dict(case_id=cid,query_id=item['query_id'],screen_status=item['screen_status']))
    output.mkdir(parents=True,exist_ok=False)
    (output/'blind-review.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=True)+'\n' for r in forms));(output/'private-mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
    manifest=dict(cases=len(forms),planned=len(forms)+missing,missing_translation_cases=missing,seed=20261124,input_manifests=hashes,
        review_sha256=sha(output/'blind-review.jsonl'),mapping_sha256=sha(output/'private-mapping.json'),code_sha256=sha(__file__),
        scope='Local screen-blind review, merging earlier 19 and remaining 53 translations. Missing translation stays absent, not negative. Reviewers do not see query IDs, cohort or model-screen status, although text may identify the source. No human judgments supplied. Original and generated text retain their own rights.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(cases=len(forms),planned=manifest['planned'],missing_translation_cases=missing,validation=check(output,output/'blind-review.jsonl'),scope=manifest['scope']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packets',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.packets,a.output)
