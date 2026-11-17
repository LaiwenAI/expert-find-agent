"""Re-read original teacher records and prove cached source lineage before new synthesis."""
import hashlib
import json
from pathlib import Path
from expert_search.corpus import build


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def run():
    root=Path('artifacts/fresh-method-data-v1');out=root/'raw-normalized'
    raw=sorted(Path('data/teachers/shenzhen').glob('*.jsonl'))
    before={str(p):sha(p) for p in raw}
    report=build('data',out)
    after={str(p):sha(p) for p in raw}
    if before!=after:raise ValueError('Original teacher inputs changed during normalization')
    matches={name:sha(out/name)==sha(Path('artifacts/corpus')/name) for name in ['experts.jsonl','evidence.jsonl']}
    if not all(matches.values()):raise ValueError('Old vector corpus no longer matches current original teacher inputs; re-encode before using it')
    lineage=dict(raw_teacher_files=before,normalization_code_sha256=sha(Path('src/expert_search/corpus.py')),
                 normalized_sha256={name:sha(out/name) for name in matches},cached_corpus_matches_raw_rebuild=matches,
                 experts=report['experts'],evidence=report['evidence'],kinds=report['kinds'],
                 scope='Actual original teacher JSONL files were re-read into a new normalization directory. Identity with the frozen vector-input corpus proven by file hashes; this is provenance validation, not semantic quality or new labels. Fresh100 demand records come directly from original XLSX via separately hashed selection.')
    (root/'raw-lineage.json').write_text(json.dumps(lineage,indent=2));print(json.dumps({k:v for k,v in lineage.items() if k not in {'raw_teacher_files','scope'}},indent=2))


if __name__=='__main__':run()
