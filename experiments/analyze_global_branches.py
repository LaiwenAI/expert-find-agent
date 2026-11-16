"""Compare source-branch versus single-root coarse recall at fixed budgets."""
import hashlib
import json
from pathlib import Path


def run():
    base=Path('artifacts/global-expert-branch-benchmark-v1');branch=json.loads((base/'summary.json').read_text())
    roots=json.loads(Path('artifacts/global-expert-root-benchmark-v1/summary.json').read_text());by_id={r['id']:r for r in roots['rows']};protocol=json.loads((base/'protocol.json').read_text())
    if {r['id'] for r in branch['rows']}!=set(by_id):raise ValueError('Changed comparison query set')
    rows=[]
    for r in branch['rows']:
        old=by_id[r['id']]
        rows.append(dict(id=r['id'],kind=r['kind'],root_known_rank=old['known_expert_rank'],branch_known_rank=r['known_expert_rank'],
            root_retention=old['retention'],branch_retention=r['retention'],root_online_seconds=old['query_encoding_seconds']+old['retrieval_seconds'],branch_online_seconds=r['query_encoding_seconds']+r['retrieval_seconds']))
    report=dict(version=1,experts=branch['experts'],branches=branch['branches'],rows=rows,
        retention={kind:{str(k):dict(root=sum(r['root_retention'][str(k)] for r in rows if r['kind']==kind),branch=sum(r['branch_retention'][str(k)] for r in rows if r['kind']==kind),queries=sum(r['kind']==kind for r in rows)) for k in protocol['budgets']} for kind in sorted({r['kind'] for r in rows})},
        mean_root_online_seconds=sum(r['root_online_seconds'] for r in rows)/len(rows),mean_branch_online_seconds=sum(r['branch_online_seconds'] for r in rows)/len(rows),
        summary_sha256=hashlib.sha256((base/'summary.json').read_bytes()).hexdigest(),
        scope='Coverage diagnostic with two reused known silver expert pairs, original queries unchanged. More source entries and changed 512-token per-source versus 1024-token root budget; not an isolated representation ablation. No precision or complete-recall claim; source verification and requirement planning excluded.')
    Path('docs/exploration/global-branch-analysis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':run()
