"""Freeze available backtranslated and Chinese views for Slurm query encoding."""
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    original=Path('artifacts/cof-sigir-bilingual-translation-v1');back=Path('artifacts/cof-sigir-backtranslation-v1')
    original_selection=json.loads((original/'encode-manifest.json').read_text());protocol=json.loads((back/'protocol.json').read_text());summary=json.loads((back/'summary.json').read_text())
    if sha(original/'encode-input.jsonl')!=protocol['input_sha256'] or sha(original/'encode-manifest.json')!=protocol['input_selection_sha256'] or sha(back/'protocol.json')!=summary['protocol_sha256']:
        raise ValueError('Changed translation artifacts')
    parents={r['id']:r for r in map(json.loads,(original/'encode-input.jsonl').read_text().splitlines())}
    if list(parents)!=protocol['query_ids']:raise ValueError('Changed original query selection')
    rows=[];missing=list(original_selection['missing_query_ids'])
    for qid in protocol['query_ids']:
        name=hashlib.sha256(qid.encode()).hexdigest()+'.json';path=back/'queries'/name
        if sha(path)!=summary['record_sha256'][name]:raise ValueError('Changed backtranslation')
        result=json.loads(path.read_text());parent=parents[qid]
        if result['query_id']!=qid or result['chinese_sha256']!=hashlib.sha256(parent['chinese'].encode()).hexdigest() or result['parent_status']!=parent['status']:
            raise ValueError('Changed source view')
        if result['backtranslation'] is None:missing.append(qid);continue
        if hashlib.sha256(result['backtranslation'].encode()).hexdigest()!=result['backtranslation_sha256']:raise ValueError('Changed generated English')
        rows.append({'id':qid,'chinese':parent['chinese'],'english':result['backtranslation'],'status':result['status'],'parent_status':parent['status'],
                     'contains_han':any('\u4e00'<=c<='\u9fff' for c in parent['chinese'])})
    if set(missing)&{r['id'] for r in rows} or set(missing)|{r['id'] for r in rows}!=set(original_selection['all_query_ids']):raise ValueError('Changed total inventory')
    target=back/'encode-input.jsonl';marker=back/'encode-manifest.json'
    if target.exists() or marker.exists():raise FileExistsError('Encoding selection exists')
    target.write_text(''.join(json.dumps(r,ensure_ascii=True)+'\n' for r in rows))
    report={'all_query_ids':original_selection['all_query_ids'],'available_query_ids':[r['id'] for r in rows],'missing_query_ids':missing,
            'available':len(rows),'model_screened_both':sum(r['status']=='model_screened' and r['parent_status']=='model_screened' for r in rows),
            'queries_sha256':sha(target),'original_selection_sha256':sha(original/'encode-manifest.json'),'backtranslation_protocol_sha256':sha(back/'protocol.json'),
            'backtranslation_summary_sha256':sha(back/'summary.json'),'checkpoint_provenance_sha256':sha('artifacts/models/qwen3-embedding-0.6b/checkpoint-provenance.json'),
            'encoder_sha256':sha('src/expert_search/qwen_embedding_torch.py'),'task_module_sha256':sha('src/expert_search/qwen_embedding.py'),
            'encoder_script_sha256':sha('experiments/bilingual/cof_query_encode.py'),
            'scope':'Chinese and backtranslated English query views; no original English text is passed to the backtranslation model. Same-model screens are not independent semantic gold.'}
    marker.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
