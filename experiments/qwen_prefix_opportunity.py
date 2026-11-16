"""Count exact common-token prefix reuse without loading model weights."""
import json
from pathlib import Path
from transformers import AutoTokenizer
from expert_search.qwen_reranker import PREFIX, SUFFIX, TASK, pack_tokens


def run():
    tokenizer = AutoTokenizer.from_pretrained('artifacts/models/qwen3-reranker-0.6b', local_files_only=True)
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    reports = []
    for dataset in ('NIPS', 'SIGIR'):
        root = Path('artifacts/public/cof')/dataset/'packing-shared-v2'
        protocol = json.loads((root/'protocol.json').read_text())
        rows = []
        for qid in protocol['all_query_ids']:
            payload = json.loads((root/f'{qid}-similarity-input.json').read_text())
            sources = {s['source_id']:s['text'] for s in payload['source_windows']}
            sequences = []
            for candidate in payload['candidates']:
                document = '\n\n'.join(sources[e['source_id']] for e in candidate['evidence'])
                body = tokenizer.encode(f"<Instruct>: {TASK}\n<Query>: {payload['target_paper']}\n<Document>: {document}", add_special_tokens=False)
                sequences.append(pack_tokens(prefix,body,suffix,4096)[0])
            common = 0
            if len(sequences)>1:
                for column in zip(*sequences):
                    if len(set(column))!=1:
                        break
                    common += 1
                common = min(common,min(map(len,sequences))-1)
            original = sum(map(len,sequences))
            saved = common*(len(sequences)-1)
            rows.append(dict(query_id=qid,candidates=len(sequences),common_prefix_tokens=common,
                             original_tokens=original,reused_tokens=saved))
        total = sum(r['original_tokens'] for r in rows)
        saved = sum(r['reused_tokens'] for r in rows)
        reports.append(dict(dataset=dataset,queries=len(rows),original_tokens=total,
                            reused_tokens=saved,fraction=saved/total,rows=rows,
                            scope='Exact token-prefix opportunity count. Runtime, KV memory, batching and numerical equivalence require measurement.'))
    Path('docs/exploration/qwen-prefix-opportunity.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k!='rows'} for r in reports],indent=2))


if __name__=='__main__':run()
