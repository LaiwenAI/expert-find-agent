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
            new=generator.ask('rewrite','用中文构造一个自然的专家研究领域搜索请求，只要求同时涉及给定的两项研究方向。可进行准确同义改写，但不添加行业、应用、部署、研发交付、经验或能力保证。不出现专家姓名、机构、引号模板或论文标题。输出 {"query":str}。',{'research_areas':base['constraints']})
            query=new.get('query')
            if not isinstance(query,str) or not query.strip():raise ValueError('Empty rewrite')
            audit=verifier.ask('equivalence','判断专家研究领域搜索请求是否恰好要求同时涉及给定两项研究方向。双向检查：有无遗漏、额外要求或概念范围改变。广泛相关不等于等义，不推断交付能力。输出 {"same_requirements":bool,"omissions":[str],"additions":[str],"scope_changes":[str],"reason":str}。',{'research_areas':base['constraints'],'query':query})
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
