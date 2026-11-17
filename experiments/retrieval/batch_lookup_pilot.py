import argparse,importlib.util,json,time,pathlib,hashlib,statistics
from expert_search.retrieval import Search
parser=argparse.ArgumentParser();parser.add_argument('--old-source',type=pathlib.Path,required=True);parser.add_argument('--output',type=pathlib.Path,required=True);args=parser.parse_args()
spec=importlib.util.spec_from_file_location('expert_search.old_retrieval',args.old_source);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
a=old.Search('artifacts/corpus','artifacts/search.sqlite');b=Search('artifacts/corpus','artifacts/search.sqlite');queries=[]
for line in open('artifacts/benchmark/runs.jsonl'):
 r=json.loads(line)
 if r['query_id'] not in {x[0] for x in queries}:queries.append((r['query_id'],r['query']))
 if len(queries)==20:break
rows=[]
for j,(qid,q) in enumerate(queries):
 results={};times={}
 for name,obj in ([('old',a),('new',b)] if j%2==0 else [('new',b),('old',a)]):
  t=time.perf_counter();results[name]=obj.search(q,20);times[name]=time.perf_counter()-t
 rows.append({'query_id':qid,'same':results['old']==results['new'],'seconds':times,'first':'old' if j%2==0 else 'new'})
p=args.output;p.mkdir(parents=True,exist_ok=False);(p/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
(p/'protocol.json').write_text(json.dumps({'old_source_sha256':hashlib.sha256(args.old_source.read_bytes()).hexdigest(),'new_source_sha256':hashlib.sha256(pathlib.Path('src/expert_search/retrieval.py').read_bytes()).hexdigest(),'benchmark_runs_sha256':hashlib.sha256(pathlib.Path('artifacts/benchmark/runs.jsonl').read_bytes()).hexdigest(),'queries':len(rows),'scope':'Alternating order, one call each; excludes initialization and warmup.'},indent=2)+'\n')
print(json.dumps({'queries':len(rows),'identical':sum(r['same'] for r in rows),'old_mean':statistics.mean(r['seconds']['old'] for r in rows),'new_mean':statistics.mean(r['seconds']['new'] for r in rows),'scope':'Alternating order, one call each, no warmup or initialization timing.'}))
