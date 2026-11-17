"""Compare a complete segmented-translation pilot with frozen direct runs."""
import json
import statistics
from collections import Counter
from pathlib import Path

from expert_search.bilingual.postvalidation import check_pair
from .translate_segmented_pilot import sha


def main():
    data=Path('artifacts/large-bilingual-data-v1');root=data/'original-translations-segmented-v1';direct=data/'original-translations-v1'
    p=json.loads((root/'protocol.json').read_text());qfile=data/'queries-zh.jsonl'
    if sha(qfile)!=p['source_sha256'] or sha(direct/'protocol.json')!=p['direct_protocol_sha256']:raise ValueError('Changed input')
    for path,digest in p['code_sha256'].items():
        if sha(Path(path))!=digest:raise ValueError('Changed code: '+path)
    parents={r['id']:r for r in map(json.loads,qfile.read_text().splitlines())};groups={'direct':[],'segmented':[]}
    for qid in p['parent_ids']:
        if parents[qid]['split']!='construction':raise ValueError('Protected query entered pilot')
        old=direct/'parents'/(qid+'.json');new=root/'parents'/(qid+'.json')
        if sha(old)!=p['direct_record_sha256'][qid] or not new.exists():raise ValueError('Changed or incomplete attempt')
        for arm,path in [('direct',old),('segmented',new)]:
            r=json.loads(path.read_text())
            if r['parent_id']!=qid or r['source_text_sha256']!=parents[qid]['source']['text_sha256']:raise ValueError('Changed parent identity')
            for seg in r['segments']:
                if parents[qid]['zh'][seg['start']:seg['end']]!=seg['zh'] or check_pair(seg['zh'],seg['en'])!=seg['numeric_check']:raise ValueError('Changed source or numeric check')
            if r['en'] is not None and (''.join(s['zh'] for s in r['segments'])!=parents[qid]['zh'] or '\n'.join(s['en'] for s in r['segments'])!=r['en']):raise ValueError('Incomplete translation assembly')
            if r['status']=='model_screened_complete_translation' and (r.get('audit',{}).get('labels',{}).get('full_translation')!='faithful_complete' or any(s['numeric_check']['status']!='match' for s in r['segments'])):raise ValueError('Invalid accepted status')
            groups[arm].append(r)
    report={}
    for arm,rows in groups.items():
        report[arm]={'attempts':len(rows),'complete_translations':sum(r['en'] is not None for r in rows),
                     'statuses':dict(Counter(r['status'] for r in rows)),
                     'native_labels':dict(Counter(r.get('audit',{}).get('labels',{}).get('full_translation','missing') for r in rows)),
                     'numeric_statuses':dict(Counter(s for r in rows for s in r.get('numeric_statuses',[]))),
                     'mean_job_seconds':statistics.mean(r['seconds'] for r in rows)}
    pairs=list(zip(groups['segmented'],groups['direct']))
    report['paired_screen_difference']=sum(int(a['status']=='model_screened_complete_translation')-int(b['status']=='model_screened_complete_translation') for a,b in pairs)
    report['segmented_observed_models']=sorted({str(a.get('model')) for r in groups['segmented'] for a in r.get('segment_attempts',[])})
    report['scope']='Sequential post-hoc comparison on 16 reused construction needs. Each arm uses gpt-6-luna, same-model Decisions review and limited numeric checks, none of which are human gold. Time includes retries=0 and multiple calls but not service initialization; concurrent job durations are not wall time. Missing attempts remain failures; no quality or general speed claim.'
    (root/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
