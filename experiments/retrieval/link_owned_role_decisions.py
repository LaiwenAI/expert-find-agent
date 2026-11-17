"""Create a private post-review mapping for native role decisions."""
import hashlib
import json
from pathlib import Path

from expert_search.owned_role_windows import source_key


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    review=Path('artifacts/owned-role-human-review-v1');decision=Path('artifacts/owned-role-decision-pilot-v1')
    parent=Path('artifacts/owned-role-paper-pilot-v1');rmanifest=json.loads((review/'manifest.json').read_text())
    protocol=json.loads((decision/'protocol.json').read_text());mapping_path=review/'private-mapping.json'
    if sha(mapping_path)!=rmanifest['mapping_sha256'] or sha(parent/'sources.json')!=protocol['source_sha256']:raise ValueError('Changed private mapping or sources')
    mapping=json.loads(mapping_path.read_text());views=json.loads((parent/'sources.json').read_text())
    indexed={(r['source_id'],r['expert_id']):r['case_id'] for r in mapping}
    if len(indexed)!=len(mapping):raise ValueError('Ambiguous review source')
    linked=[];attempt_hashes={}
    for i,source in enumerate(views,1):
        path=decision/(f'source-{i:03d}.json');r=json.loads(path.read_text());attempt_hashes[path.name]=sha(path)
        if r['source_key']!=source_key(source,protocol['width']):raise ValueError('Changed source binding')
        key=(source['id'],source['expert_id'])
        if key not in indexed:raise ValueError('No matching review case')
        linked.append({'case_id':indexed[key],'status':r['status'],'role_record':r['role_record']})
    output=decision/'private-review-link.json'
    if output.exists():raise FileExistsError('Preserve existing mapping')
    value={'links':linked,'review_manifest_sha256':sha(review/'manifest.json'),'decision_protocol_sha256':sha(decision/'protocol.json'),
           'decision_attempt_sha256':attempt_hashes,'scope':'Private post-review mapping; do not show to reviewers before their judgments.'}
    output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'linked_sources':len(linked),'sha256':sha(output)}))


if __name__=='__main__':main()
