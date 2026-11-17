"""Refresh manuscript numbers from authoritative artifacts without API calls."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path


def export():
    root=Path(__file__).resolve().parents[2];out=root/'docs/paper/sigir/generated';out.mkdir(exist_ok=True)
    paths=dict(merit='artifacts/merit-suitability-comparison-v1/evaluation.json',large='artifacts/large-bilingual-data-v1/summary.json',plan='artifacts/large-bilingual-data-v1/synthesis-v1/protocol.json',metadata='artifacts/source-metadata-v1/manifest.json',industrial='artifacts/dataset-release/fresh-enterprise-eval-v1/manifest.json',postcheck='artifacts/large-bilingual-data-v1/postvalidation-v2/manifest.json')
    data={k:json.loads((root/v).read_text()) for k,v in paths.items()};planned=data['plan']['selected_parent_ids'];rows=[];file_hashes={}
    directory=root/'artifacts/large-bilingual-data-v1/synthesis-v1/parents'
    for qid in planned:
        p=directory/(qid+'.json')
        if not p.exists():continue
        row=json.loads(p.read_text())
        if row['parent_id']!=qid or row['status'] not in ('unresolved','completed_with_unjudged_retained'):raise ValueError('Invalid terminal synthesis record')
        rows.append(row);file_hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    generated=sum(len(r['variants']) for r in rows);screened=sum(v.get('label_status')=='model_screened_query_pair' for r in rows for v in r['variants'].values())
    values=dict(RealLarge=data['large']['new_original_demands'],RealSelected=len(planned),SyntheticPlanned=data['plan']['planned_synthetic_intents'],ParentJobsDone=len(rows),SyntheticGenerated=generated,SyntheticScreened=screened,SyntheticUnjudged=generated-screened,ExpertFitLabelsLarge=0,SnapshotUTC=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'))
    post=data['postcheck'];values.update(PostPairs=post['generated_pairs'],PostRetained=post['counts']['retained_screened'],PostDowngraded=post['counts']['downgraded'],PostUncertain=post['counts']['uncertain'],PostSnapshotUTC=post['snapshot_utc'][:16].replace('T',' ')+' UTC')
    for arm,key in [('official-long','Long'),('short-chat','Short'),('native-choice','Native')]:
        a=data['merit']['arms'][arm]
        for metric,suffix in [('accuracy','Acc'),('precision','Prec'),('recall','Recall'),('f1','Fone'),('balanced_accuracy','BAcc')]:values[key+suffix]=f"{100*a['valid_only'][metric]:.2f}"
        values[key+'Seconds']=f"{a['mean_seconds']:.2f}";values[key+'Pninetyfive']=f"{a['empirical_p95_seconds']:.2f}"
    (out/'results.tex').write_text('% Generated from immutable results plus terminal synthesis records.\n'+''.join('\\newcommand{\\'+k+'}{'+str(v)+'}\n' for k,v in values.items()))
    manifest=dict(values=values,inputs_sha256={v:hashlib.sha256((root/v).read_bytes()).hexdigest() for v in paths.values()},synthesis_terminal_records_sha256=file_hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Counts are a timestamped snapshot, not a live-process claim. Synthetic screen status is same-model diagnostic, not expert-fit/human gold; source language views not independent queries.')
    (out/'results.json').write_text(json.dumps(manifest,indent=2)+'\n');return values


if __name__=='__main__':print(json.dumps(export(),indent=2))
