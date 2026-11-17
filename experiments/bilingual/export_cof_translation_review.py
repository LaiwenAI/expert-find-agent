"""Prepare private, screen-blind review of available CoF query translations."""
import argparse
import hashlib
import json
from pathlib import Path
import random


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(folder,output):
    selection=json.loads((folder/'encode-manifest.json').read_text());summary=json.loads((folder/'summary.json').read_text())
    if sha(folder/'encode-input.jsonl')!=selection['queries_sha256'] or sha(folder/'protocol.json')!=selection['translation_protocol_sha256'] or sha(folder/'summary.json')!=selection['translation_summary_sha256']:
        raise ValueError('Changed frozen translation inputs')
    rows=[json.loads(line) for line in (folder/'encode-input.jsonl').read_text().splitlines()]
    if [r['id'] for r in rows]!=selection['available_query_ids'] or len({r['id'] for r in rows})!=len(rows):raise ValueError('Changed query inventory')
    for row in rows:
        name=hashlib.sha256(row['id'].encode()).hexdigest()+'.json';path=folder/'queries'/name
        if sha(path)!=summary['record_sha256'][name]:raise ValueError('Changed translation record')
        original=json.loads(path.read_text())
        if original['query_id']!=row['id'] or original['chinese']!=row['chinese'] or original['status']!=row['status'] or hashlib.sha256(row['english'].encode()).hexdigest()!=original['english_sha256']:
            raise ValueError('Changed bilingual view')
    random.Random(20261123).shuffle(rows);forms=[];mapping=[]
    for index,row in enumerate(rows,1):
        case=f'case-{index:03d}';forms.append({'case_id':case,'english_source':row['english'],'chinese_translation':row['chinese'],
                                               'equivalence':None,'rationale':None,'reviewer_id':None})
        mapping.append({'case_id':case,'query_id':row['id'],'screen_status':row['status']})
    output.mkdir(parents=True,exist_ok=False);payload=''.join(json.dumps(r,ensure_ascii=True)+'\n' for r in forms).encode();private=(json.dumps(mapping,indent=2)+'\n').encode()
    (output/'blind-review.jsonl').write_bytes(payload);(output/'private-mapping.json').write_bytes(private)
    manifest={'cases':len(forms),'missing_translation_cases':len(selection['missing_query_ids']),'seed':20261123,
              'selection_sha256':sha(folder/'encode-manifest.json'),'translation_summary_sha256':sha(folder/'summary.json'),
              'review_sha256':hashlib.sha256(payload).hexdigest(),'mapping_sha256':hashlib.sha256(private).hexdigest(),'code_sha256':sha(Path(__file__)),
              'scope':'Local screen-blind translation review of previously studied public queries; IDs and model-screen status are hidden but text may reveal source. Missing translations remain absent, not negative. No human labels are provided.'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'cases':len(forms),'missing_translation_cases':manifest['missing_translation_cases'],'scope':manifest['scope']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.folder,a.output)
