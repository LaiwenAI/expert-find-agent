"""Create a source-only, model-output-blind SciFact review packet."""
import argparse
import hashlib
import json
from pathlib import Path
from experiments.scifact.decisions_oracle_pilot import load,sha


def main(archive,parent,output):
    protocol=json.loads((parent/'protocol.json').read_text());summary=json.loads((parent/'summary.json').read_text())
    if sha(archive)!=protocol['archive_sha256'] or sha(parent/'protocol.json')!=summary['protocol_sha256'] or sha(parent/'results.json')!=summary['results_sha256']:
        raise ValueError('Changed source or pilot records')
    corpus,claims=load(archive);papers={str(p['doc_id']):p for p in corpus};by_claim={c['id']:c for c in claims}
    output.mkdir(parents=True,exist_ok=False);views=[];mapping=[]
    for i,item in enumerate(protocol['cases']):
        c=by_claim[item['claim_id']];p=papers[item['doc_id']];key='case-'+str(i+1).zfill(3)
        if item['doc_id'] in c['evidence'] or int(item['doc_id']) not in c['cited_doc_ids']:raise ValueError('Changed selection')
        view={'case_id':key,'claim':c['claim'],'title':p['title'],'abstract':p['abstract']}
        view_hash=hashlib.sha256(json.dumps(view,sort_keys=True,ensure_ascii=True).encode()).hexdigest()
        views.append(dict(**view,view_sha256=view_hash,verdict=None,evidence_quote=None,reviewer_id=None,notes=None))
        mapping.append({'case_id':key,'claim_id':item['claim_id'],'doc_id':item['doc_id'],'view_sha256':view_hash})
    (output/'review.json').write_text(json.dumps(views,ensure_ascii=True,indent=2)+'\n')
    (output/'private-mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
    report={'cases':len(views),'source_sha256':sha(archive),'protocol_sha256':sha(parent/'protocol.json'),
            'initial_review_sha256':sha(output/'review.json'),'mapping_sha256':sha(output/'private-mapping.json'),
            'scope':'Source-only local packet; model outputs and annotation status hidden. Source text can reveal identity. Null judgments are not negatives. Review is not yet completed.'}
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.archive,a.parent,a.output)
