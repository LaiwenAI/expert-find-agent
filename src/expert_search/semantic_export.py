"""Export audited semantic diagnostics without original source text or model traces."""
import hashlib
import json
from pathlib import Path


def export(output='artifacts/dataset-release/semantic-diagnostic'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);rows=[];rejections=[]
    for p in sorted(Path('artifacts/semantic-constraints').glob('conjunction-*.json')):
        r=json.loads(p.read_text())
        if r['status']!='audited_silver':rejections.append(r['id']);continue
        family_hash=hashlib.sha256(r['construction_expert'].encode()).hexdigest()
        rows.append(dict(id=r['id'],query=r['query'],constraints=r['base_constraints'],known_source_experts=r['known_source_experts'],
            references=r['references'],family_group=family_hash,label_quality='model-audited silver; incomplete relevance labels',
            task='semantic research-area conjunction diagnostic',source_record_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (out/'examples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    manifest=dict(examples=len(rows),rejected_ids=rejections,source_text_included=False,model_traces_included=False,
        annotation_note='Model equivalence audit, not human expertise relevance. Does not replace industrial demand dataset.',
        redistribution_note='Local release candidate only; source and derivative rights review required before public publication.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest

if __name__=='__main__':print(json.dumps(export(),indent=2))
