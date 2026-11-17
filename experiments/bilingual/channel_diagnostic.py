"""Fixed-score source-channel/margin controls; CPU Slurm computation only."""
import hashlib
import json
from pathlib import Path
import platform
import time
from collections import Counter
import numpy as np

PARENT = Path('artifacts/bilingual-retrieval-v1')
ROOT = Path('artifacts/bilingual-channel-diagnostic-v1')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run():
    parent = json.loads((PARENT / 'protocol.json').read_text())
    summary = json.loads((PARENT / 'summary.json').read_text())
    source = Path('artifacts/global-expert-branches-v1/branches.jsonl')
    vp = Path('artifacts/global-expert-branch-benchmark-v1/vectors.npy')
    if sha(source) != parent['source_sha256'] or sha(vp) != parent['source_vectors_sha256'] or sha(PARENT / 'query-vectors.npz') != summary['encoding']['query_vectors_sha256'] or sha(PARENT / 'evaluation.jsonl') != summary['evaluation_sha256']:
        raise ValueError('Changed frozen source/query/rank data')
    protocol = dict(source_sha256=sha(source), vectors_sha256=sha(vp), parent_protocol_sha256=sha(PARENT / 'protocol.json'), queries_sha256=sha(PARENT / 'query-vectors.npz'), parent_evaluation_sha256=sha(PARENT / 'evaluation.jsonl'), channels=['all', 'paper', 'profile', 'all_common', 'paper_common', 'profile_common'], k=10, common_population='Experts with at least one paper and one profile source; identical expert IDs in all_common/paper_common/profile_common.', score='Same FP32 einsum maximum-source score and lexicographic expert tie handling as parent exact reference.', code_sha256=sha(Path(__file__)), scope='Source-channel intervention on frozen scores, not semantic accuracy. Channel restriction can change both evidence and eligible experts; common variants control expert IDs but not evidence. Rank margins are numerical sensitivity diagnostics, not causal proof or capability judgments. No LLM calls, new embeddings or model training.')
    ROOT.mkdir(exist_ok=True)
    marker = ROOT / 'protocol.json'
    if marker.exists() and json.loads(marker.read_text()) != protocol:
        raise ValueError('Changed channel diagnostic protocol')
    marker.write_text(json.dumps(protocol, indent=2) + '\n')
    records = list(map(json.loads, source.read_text().splitlines()))
    matrix = np.load(vp, mmap_mode='r', allow_pickle=False)
    with np.load(PARENT / 'query-vectors.npz', allow_pickle=False) as cache:
        vectors, ids = cache['vectors'], list(cache['ids'])
    references = list(map(json.loads, (PARENT / 'evaluation.jsonl').read_text().splitlines()))
    if len(references) != 1000 or ids != [r['id'] for r in references]:
        raise ValueError('Changed paired query order')
    all_owners, paper_owners, profile_owners = {}, {}, {}
    for index, row in enumerate(records):
        all_owners.setdefault(row['expert_id'], []).append(index)
        (paper_owners if row['kind'] == 'paper' else profile_owners).setdefault(row['expert_id'], []).append(index)
    common = paper_owners.keys() & profile_owners.keys()
    owner_sets = dict(all=all_owners, paper=paper_owners, profile=profile_owners)
    owner_sets.update({name+'_common': {key: value for key, value in rows.items() if key in common} for name, rows in list(owner_sets.items())})
    channels = {}
    for name, owners in owner_sets.items():
        expert_ids = sorted(owners)
        counts = np.asarray([len(owners[e]) for e in expert_ids])
        channels[name] = dict(expert_ids=expert_ids, owners=owners, flat=np.asarray([i for e in expert_ids for i in owners[e]],dtype=np.int64), starts=np.concatenate(([0],np.cumsum(counts)[:-1])))
    def rank(scores, channel):
        c = channels[channel]
        maxima = np.maximum.reduceat(scores[c['flat']], c['starts'])
        order = np.argsort(-maxima,kind='stable')
        ranking = [c['expert_ids'][i] for i in order[:20]]
        top_scores = [float(maxima[i]) for i in order[:20]]
        source_ids = [min(c['owners'][e],key=lambda i:(-float(scores[i]),records[i]['id'])) for e in ranking[:10]]
        ranks = {c['expert_ids'][i]: position+1 for position,i in enumerate(order)}
        return dict(ranking=ranking, scores=top_scores, winning_source_ids=[records[i]['id'] for i in source_ids], paper_fraction=sum(records[i]['kind']=='paper' for i in source_ids)/10, distinct_prefixes=len({records[i]['source_prefix_sha256'] for i in source_ids}), cutoff_gap=top_scores[9]-top_scores[10]),ranks
    output = ROOT / 'evaluation.jsonl'
    rows = []
    start = time.perf_counter()
    with output.open('w') as stream:
        for qi, reference in enumerate(references):
            row = {key: reference[key] for key in ['id','parent_id','group_id','intent_type','status']}
            results, full_ranks = {}, {}
            for li, language in enumerate(['zh','en']):
                scores = np.einsum('ij,j->i',matrix,vectors[2*qi+li],optimize=False)
                results[language], full_ranks[language] = {}, {}
                for name in channels:
                    results[language][name], full_ranks[language][name] = rank(scores,name)
                if results[language]['all']['ranking'][:10] != reference['languages'][language]['exact']['ranking']:
                    raise ValueError('Parent exact-ranking reproduction failed')
            row['channels'] = {}
            for name in channels:
                zh,en = results['zh'][name],results['en'][name]
                row['channels'][name] = dict(languages=dict(zh=zh,en=en), overlap=len(set(zh['ranking'][:10])&set(en['ranking'][:10]))/10, zh_top10_ranks_in_en=[full_ranks['en'][name][eid] for eid in zh['ranking'][:10]], en_top10_ranks_in_zh=[full_ranks['zh'][name][eid] for eid in en['ranking'][:10]])
            stream.write(json.dumps(row)+'\n');stream.flush();rows.append(row)
            if (qi+1)%100==0:print('Channel pairs',qi+1,flush=True)
    def aggregate(subset):
        return {name: dict(pairs=len(subset),eligible_experts=len(channels[name]['expert_ids']),mean_overlap=float(np.mean([r['channels'][name]['overlap'] for r in subset])),languages={lang:dict(mean_paper_fraction=float(np.mean([r['channels'][name]['languages'][lang]['paper_fraction'] for r in subset])),median_cutoff_gap=float(np.median([r['channels'][name]['languages'][lang]['cutoff_gap'] for r in subset])),zero_cutoff_gaps=sum(r['channels'][name]['languages'][lang]['cutoff_gap']==0 for r in subset),mean_distinct_prefixes=float(np.mean([r['channels'][name]['languages'][lang]['distinct_prefixes'] for r in subset]))) for lang in ['zh','en']}) for name in channels}
    result = dict(pairs=len(rows),all=aggregate(rows),by_status={status:aggregate([r for r in rows if r['status']==status]) for status in sorted({r['status'] for r in rows})},source_prefix_unique=len({r['source_prefix_sha256'] for r in records}),source_rows=len(records),wall_seconds=time.perf_counter()-start,protocol_sha256=sha(marker),evaluation_sha256=sha(output),node=platform.node(),scope=protocol['scope'])
    (ROOT / 'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['all'],indent=2))


if __name__ == '__main__':
    run()
