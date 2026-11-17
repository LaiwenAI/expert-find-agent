"""Build temporal/bibliometric sidecar from original papers without changing vectors."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from expert_search.metadata.papers import paper_metadata,availability


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def run():
    source=Path('data/teachers/shenzhen/papers.jsonl');out=Path('artifacts/source-metadata-v1');out.mkdir(exist_ok=True);rows=[];counts=Counter();publication_cutoff=Counter()
    with source.open() as f:
        for line,text in enumerate(f,1):
            raw=json.loads(text);meta=paper_metadata(raw);row=dict(source_id='paper:'+str(line),expert_id=raw['teacher_id'],raw_line=line,raw_record_sha256=hashlib.sha256(json.dumps(raw,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),**meta);rows.append(row)
            counts['papers']+=1;counts['date_'+meta['publication']['status']]+=1
            if meta['publication']['precision']:counts['date_precision_'+meta['publication']['precision']]+=1
            counts['doi_present']+=bool(meta['doi']);counts['author_list_present']+=bool(meta['author_list_text']);counts['citation_total_known']+=meta['citations']['observed_total'] is not None;counts['duplicate_citation_years']+=meta['citations']['duplicate_years'];counts['invalid_citation_history_entries']+=meta['citations']['invalid_history_entries'];counts['citation_'+meta['citations']['total_history_status']]+=1
            publication_cutoff[availability(meta['publication'],'2025-12-31')]+=1
    path=out/'papers.jsonl';body=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    if path.exists() and path.read_text()!=body:raise ValueError('Changed metadata sidecar')
    path.write_text(body)
    manifest=dict(source_file=str(source),source_sha256=sha(source),rows=len(rows),metadata_sha256=sha(path),code_sha256={str(p):sha(p) for p in [Path(__file__),Path('src/expert_search/metadata/papers.py')]},counts=dict(counts),publication_availability_as_of_2025_12_31=dict(publication_cutoff),explicit_personal_contribution_roles_available=0,scope='Original paper metadata sidecar only, original source text/index inputs unchanged. Publication intervals retain date precision; citation observation time unknown. No personal-credit score, honorary-authorship inference, citation prestige rank or temporal accuracy claim. Role/ORCID/source verification remains necessary.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))


if __name__=='__main__':run()
