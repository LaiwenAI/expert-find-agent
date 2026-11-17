"""Audited paraphrases of source-linked constraints; open-world known positives only."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from laiwenai import LaiwenAI
from .agent import ExpertAgent
from .corpus import records


def run(n=50,workers=4,output='artifacts/semantic-constraints'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    bases=[r for _,r in records('artifacts/controlled-constraints/queries.jsonl')][:n]
    def one(base):
        p=out/f"{base['id']}.json"
        if p.exists():return json.loads(p.read_text())['status']
        generator=ExpertAgent(client=LaiwenAI(model='gpt-4o'))
        verifier=ExpertAgent(client=LaiwenAI(model='gpt-6-luna'))
        try:
            new=generator.ask('rewrite','\u7528\u4e2d\u6587\u6784\u9020\u4e00\u4e2a\u81ea\u7136\u7684\u4e13\u5bb6\u7814\u7a76\u9886\u57df\u641c\u7d22\u8bf7\u6c42\uff0c\u53ea\u8981\u6c42\u540c\u65f6\u6d89\u53ca\u7ed9\u5b9a\u7684\u4e24\u9879\u7814\u7a76\u65b9\u5411\u3002\u53ef\u8fdb\u884c\u51c6\u786e\u540c\u4e49\u6539\u5199\uff0c\u4f46\u4e0d\u6dfb\u52a0\u884c\u4e1a\u3001\u5e94\u7528\u3001\u90e8\u7f72\u3001\u7814\u53d1\u4ea4\u4ed8\u3001\u7ecf\u9a8c\u6216\u80fd\u529b\u4fdd\u8bc1\u3002\u4e0d\u51fa\u73b0\u4e13\u5bb6\u59d3\u540d\u3001\u673a\u6784\u3001\u5f15\u53f7\u6a21\u677f\u6216\u8bba\u6587\u6807\u9898\u3002\u8f93\u51fa {"query":str}\u3002',{'research_areas':base['constraints']})
            query=new.get('query')
            if not isinstance(query,str) or not query.strip():raise ValueError('Empty rewrite')
            audit=verifier.ask('equivalence','\u5224\u65ad\u4e13\u5bb6\u7814\u7a76\u9886\u57df\u641c\u7d22\u8bf7\u6c42\u662f\u5426\u6070\u597d\u8981\u6c42\u540c\u65f6\u6d89\u53ca\u7ed9\u5b9a\u4e24\u9879\u7814\u7a76\u65b9\u5411\u3002\u53cc\u5411\u68c0\u67e5\uff1a\u6709\u65e0\u9057\u6f0f\u3001\u989d\u5916\u8981\u6c42\u6216\u6982\u5ff5\u8303\u56f4\u6539\u53d8\u3002\u5e7f\u6cdb\u76f8\u5173\u4e0d\u7b49\u4e8e\u7b49\u4e49\uff0c\u4e0d\u63a8\u65ad\u4ea4\u4ed8\u80fd\u529b\u3002\u8f93\u51fa {"same_requirements":bool,"omissions":[str],"additions":[str],"scope_changes":[str],"reason":str}\u3002',{'research_areas':base['constraints'],'query':query})
            if not isinstance(audit.get('same_requirements'),bool):raise ValueError('Invalid audit')
            accepted=audit['same_requirements'] and all(isinstance(audit.get(k),list) and not audit[k] for k in ['omissions','additions','scope_changes'])
            row=dict(status='audited_silver' if accepted else 'rejected',id=base['id'],query=query,base_constraints=base['constraints'],
                known_source_experts=base['known_both'],construction_expert=base['construction_expert'],references=base['references'],audit=audit,
                generator_trace=generator.trace,verifier_trace=verifier.trace,
                label_semantics='Known sources only. Other experts may also be relevant. Model equivalence audit is not human gold.')
        except Exception as e:row=dict(status='error',id=base['id'],error=f'{type(e).__name__}: {e}')
        p.write_text(json.dumps(row,ensure_ascii=False,indent=2));return row['status']
    with ThreadPoolExecutor(max_workers=workers) as pool:statuses=list(pool.map(one,bases))
    summary=dict(attempts=len(statuses),accepted=statuses.count('audited_silver'),rejected=statuses.count('rejected'),errors=statuses.count('error'),
        warning='Semantic conjunction diagnostic; not actual enterprise task gold and not complete qrels.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2));return summary

if __name__=='__main__':print(run())
