"""Diagnose cache-state mutation and segmented-forward precision on fixed inputs."""
import json
from pathlib import Path
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from expert_search.qwen_prefix_cache import PrefixCachedReranker


def run():
    mx.set_memory_limit(4*1024**3)
    mx.set_cache_limit(128*1024**2)
    model=PrefixCachedReranker()
    query='robot motion control and trajectory planning'
    documents=['A study of robot motion control and planning.',
               'Marine biology and ocean ecosystems.',
               'Robot trajectory optimization with obstacle avoidance.']
    results=[]
    for dtype in (mx.bfloat16,mx.float32):
        model.model.apply(lambda value:value.astype(dtype))
        model.weight=model.model.model.embed_tokens.weight
        model.answer_weight=model.weight[mx.array(model.answer_ids)].astype(mx.float32)
        mx.eval(model.model.parameters(),model.answer_weight)
        full=model.score(query,documents)['scores']
        first=model.score_cached(query,documents)['scores']
        second=model.score_cached(query,documents)['scores']
        results.append(dict(dtype=str(dtype),full=full,cached=first,
            maximum_difference=max(abs(a-b) for a,b in zip(full,first)),
            repeated_cache_difference=max(abs(a-b) for a,b in zip(first,second))))
    report=dict(inputs='Same fixed three-document smoke case',results=results,
                scope='Precision diagnostic only; FP32 execution changes arithmetic and requires its own quality and latency evaluation.')
    Path('docs/exploration/qwen-prefix-precision.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':run()
