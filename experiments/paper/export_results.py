"""Refresh manuscript numbers from authoritative artifacts without API calls."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path


def export():
    root=Path(__file__).resolve().parents[2];out=root/'docs/paper/sigir/generated';out.mkdir(exist_ok=True)
    paths=dict(merit='artifacts/merit-suitability-comparison-v1/evaluation.json',large='artifacts/large-bilingual-data-v1/summary.json',plan='artifacts/large-bilingual-data-v1/synthesis-v1/protocol.json',metadata='artifacts/source-metadata-v1/manifest.json',industrial='artifacts/dataset-release/fresh-enterprise-eval-v1/manifest.json',postcheck='artifacts/large-bilingual-data-v1/postvalidation-v2/manifest.json',bilingual='artifacts/bilingual-retrieval-v1/summary.json',translation='artifacts/large-bilingual-data-v1/translation-comparison-v1/summary.json',factored='artifacts/merit-factored-comparison-v1/evaluation.json')
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
    bilingual=data['bilingual'];values.update(BilingualPairs=bilingual['all']['pairs'],BilingualScreenedPairs=bilingual['by_status']['retained_model_screened_pair']['pairs'],BilingualExactOverlap=f"{100*bilingual['all']['methods']['exact']['mean_language_top10_overlap']:.2f}",BilingualScreenedOverlap=f"{100*bilingual['by_status']['retained_model_screened_pair']['methods']['exact']['mean_language_top10_overlap']:.2f}",BilingualBuildSeconds=f"{bilingual['index_build_and_copies_seconds']:.2f}",BilingualEncodingSeconds=f"{bilingual['encoding']['batch_encoding_wall_seconds']:.2f}")
    for arm,key in [('exact','BExact'),('ef64','BLow'),('ef512','BHigh')]:
        metrics=bilingual['all']['methods'][arm];values[key+'Millis']=f"{1000*metrics['mean_query_seconds']:.2f}";values[key+'Fidelity']=f"{100*metrics['mean_exact_top10_retained']:.2f}";values[key+'Ordered']=metrics['ordered_fidelity']
    values['TranslationPilot']=data['translation']['pairs']
    for arm,key in [('direct','Direct'),('placeholders','Placeholder'),('fixed_blocks','Blocks')]:
        metrics=data['translation']['arms'][arm];values[key+'Pass']=metrics['statuses'].get('model_screened_complete_translation',0);values[key+'Failures']=metrics['statuses'].get('unresolved',0);values[key+'Seconds']=f"{metrics['mean_job_seconds']:.2f}"
    for arm,key in [('native-binary','CurrentBinary'),('native-factored','Factored')]:
        metrics=data['factored']['arms'][arm];values[key+'Valid']=metrics['valid'];values[key+'Seconds']=f"{metrics['mean_seconds']:.2f}"
        for metric,suffix in [('accuracy','Acc'),('precision','Prec'),('recall','Recall')]:values[key+suffix]=f"{100*metrics['valid_only'][metric]:.2f}"
        for bound,suffix in [(0,'Lower'),(1,'Upper')]:values[key+'Acc'+suffix]=f"{100*metrics['full_cohort_bounds']['accuracy'][bound]:.2f}"
    for arm,key in [('official-long','Long'),('short-chat','Short'),('native-choice','Native')]:
        a=data['merit']['arms'][arm]
        for metric,suffix in [('accuracy','Acc'),('precision','Prec'),('recall','Recall'),('f1','Fone'),('balanced_accuracy','BAcc')]:values[key+suffix]=f"{100*a['valid_only'][metric]:.2f}"
        values[key+'Seconds']=f"{a['mean_seconds']:.2f}";values[key+'Pninetyfive']=f"{a['empirical_p95_seconds']:.2f}"
    (out/'results.tex').write_text('% Generated from immutable results plus terminal synthesis records.\n'+''.join('\\newcommand{\\'+k+'}{'+str(v)+'}\n' for k,v in values.items()))
    manifest=dict(values=values,inputs_sha256={v:hashlib.sha256((root/v).read_bytes()).hexdigest() for v in paths.values()},synthesis_terminal_records_sha256=file_hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Counts are a timestamped snapshot, not a live-process claim. Synthetic screen status is same-model diagnostic, not expert-fit/human gold; source language views not independent queries.')
    (out/'results.json').write_text(json.dumps(manifest,indent=2)+'\n');return values


if __name__=='__main__':print(json.dumps(export(),indent=2))
