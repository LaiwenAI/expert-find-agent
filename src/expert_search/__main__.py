import argparse
import json
from .corpus import build

p=argparse.ArgumentParser(description='Expert search research CLI')
s=p.add_subparsers(dest='command',required=True)
b=s.add_parser('prepare'); b.add_argument('--data',default='data'); b.add_argument('--output',default='artifacts/corpus')
i=s.add_parser('index'); i.add_argument('--corpus',default='artifacts/corpus'); i.add_argument('--database',default='artifacts/search.sqlite')
q=s.add_parser('search'); q.add_argument('query'); q.add_argument('--corpus',default='artifacts/corpus'); q.add_argument('--database',default='artifacts/search.sqlite'); q.add_argument('--method',choices=['bm25-max','bm25-vote','tfidf-max'],default='bm25-max'); q.add_argument('-k',type=int,default=10)
g=s.add_parser('agent'); g.add_argument('query'); g.add_argument('-k',type=int,default=5); g.add_argument('--candidates',type=int,default=20)
s.add_parser('benchmark')
s.add_parser('demands')
s.add_parser('agent-benchmark')
s.add_parser('report')
s.add_parser('verify')
s.add_parser('demand-cases')
f=s.add_parser('fast-agent'); f.add_argument('query'); f.add_argument('-k',type=int,default=5); f.add_argument('--explain',action='store_true'); f.add_argument('--full-evidence',action='store_true'); f.add_argument('--reasoning-effort',choices=['none','low','medium','high'])
a=p.parse_args()
if a.command=='prepare':
    print(json.dumps(build(a.data,a.output),ensure_ascii=False,indent=2))

elif a.command=='index':
    from .retrieval import index
    index(a.corpus,a.database)
elif a.command=='search':
    from .retrieval import Search
    print(json.dumps(Search(a.corpus,a.database).search(a.query,a.k,a.method),ensure_ascii=False,indent=2))

elif a.command=='agent':
    from .agent import ExpertAgent
    print(json.dumps(ExpertAgent().run(a.query,a.k,a.candidates),ensure_ascii=False,indent=2))

elif a.command=='benchmark':
    from .benchmark import run
    print(json.dumps(run(),ensure_ascii=False,indent=2))

elif a.command=='demands':
    from .demands import export,pool
    print(json.dumps(pool(export()),ensure_ascii=False))

elif a.command=='agent-benchmark':
    from .agent_benchmark import run
    print(json.dumps(run(),ensure_ascii=False,indent=2))

elif a.command=='report':
    from .report import generate
    print(generate())

elif a.command=='verify':
    from .verify import verify
    print(json.dumps(verify()))
elif a.command=='demand-cases':
    from .demand_cases import run
    print(run())

elif a.command=='fast-agent':
    from .fast_agent import FastExpertAgent
    for event in FastExpertAgent(full_database='artifacts/fast-search.sqlite' if a.full_evidence else None,reasoning_effort=a.reasoning_effort).events(a.query,a.k,a.explain):
        print(json.dumps(event,ensure_ascii=False),flush=True)
