"""Encode frozen public CoF bilingual queries only on a Slurm GPU node."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main(root):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Require a Slurm compute allocation')
    root=Path(root);selection=json.loads((root/'encode-manifest.json').read_text());source=root/'encode-input.jsonl'
    if sha(source)!=selection['queries_sha256'] or sha(__file__)!=selection['encoder_script_sha256'] or sha('src/expert_search/qwen_embedding_torch.py')!=selection['encoder_sha256'] or sha('src/expert_search/qwen_embedding.py')!=selection['task_module_sha256']:
        raise ValueError('Changed encoding inputs or implementation')
    rows=[json.loads(line) for line in source.read_text().splitlines()]
    if [r['id'] for r in rows]!=selection['available_query_ids']:raise ValueError('Changed query selection')
    model_dir=Path('artifacts/models/qwen3-embedding-0.6b');provenance_path=model_dir/'checkpoint-provenance.json'
    if sha(provenance_path)!=selection['checkpoint_provenance_sha256']:raise ValueError('Changed checkpoint provenance')
    provenance=json.loads(provenance_path.read_text())
    for name,digest in provenance['files'].items():
        if sha(model_dir/name)!=digest:raise ValueError('Changed official checkpoint')
    import torch
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from expert_search.qwen_embedding import TASK
    if not torch.cuda.is_available():raise RuntimeError('GPU allocation required')
    start=time.perf_counter();model=QwenTorchEmbedding(max_length=512,batch_size=8);init=time.perf_counter()-start
    texts=[r[language] for r in rows for language in ('english','chinese')]
    lengths=[len(model.tokenizer.encode('Instruct: '+TASK+'\nQuery:'+text,add_special_tokens=True)) for text in texts]
    eligible=[i for i in range(len(rows)) if lengths[2*i]<=512 and lengths[2*i+1]<=512 and rows[i]['contains_han']]
    selected=[texts[2*i+j] for i in eligible for j in (0,1)]
    if not selected:raise ValueError('No complete nontruncated bilingual pairs')
    start=time.perf_counter();vectors=model.encode(selected,query=True);seconds=time.perf_counter()-start
    if vectors.shape!=(2*len(eligible),1024) or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5):raise ValueError('Invalid query vectors')
    out=root/'query-vectors.npz'
    if out.exists():raise FileExistsError('Do not overwrite encoding')
    np.savez(out,vectors=vectors,ids=np.asarray([rows[i]['id'] for i in eligible]),token_lengths=np.asarray(lengths,dtype=np.int32),eligible_indices=np.asarray(eligible,dtype=np.int32))
    report={'query_pairs_input':len(rows),'encoded_pairs':len(eligible),'excluded_pairs':len(rows)-len(eligible),'token_lengths':lengths,
            'initialization_seconds':init,'batch_encoding_seconds':seconds,'gpu':torch.cuda.get_device_name(0),'node':platform.node(),
            'slurm_job_id':os.environ['SLURM_JOB_ID'],'queries_sha256':sha(source),'selection_sha256':sha(root/'encode-manifest.json'),
            'checkpoint_revision':provenance['revision'],'checkpoint_provenance_sha256':sha(provenance_path),'vectors_sha256':sha(out),
            'scope':'Official frozen Qwen3 embedding; paired views are never truncated. Batch timing is throughput, not single-query service latency. Query translation and reviewer-label quality remain unverified.'}
    (root/'encoding-summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',type=Path,default=Path('artifacts/cof-sigir-bilingual-translation-v1'));a=p.parse_args();main(a.folder)
