"""Shared causal prefix with independent per-document KV branches."""
from .qwen_reranker import QwenReranker


class PrefixCachedReranker(QwenReranker):
    def score_cached(self, query, documents):
        from mlx_lm.models.cache import make_prompt_cache, KVCache
        mx = self.mx
        sequences = [self.encode(query, document)[0] for document in documents]
        if not sequences:
            return dict(scores=[], common_prefix_tokens=0)
        common = 0
        for column in zip(*sequences):
            if len(set(column)) != 1:
                break
            common += 1
        common = min(common, min(map(len, sequences))-1)
        if len(sequences) == 1 or common == 0:
            return dict(scores=self._batch(sequences), common_prefix_tokens=0)
        shared = make_prompt_cache(self.model)
        hidden = self.model.model(mx.array([sequences[0][:common]]), cache=shared)
        mx.eval(hidden, [cache.state for cache in shared])
        scores = [None]*len(sequences)
        order = sorted(range(len(sequences)), key=lambda i: (len(sequences[i]), i))
        for start in range(0, len(order), self.batch_size):
            selected = order[start:start+self.batch_size]
            branches = []
            for cache in shared:
                branch = KVCache()
                keys, values, offset = cache.state
                # Materialize independent arrays because KVCache updates in place.
                branch.state = (mx.repeat(keys[..., :offset, :], len(selected), axis=0),
                                mx.repeat(values[..., :offset, :], len(selected), axis=0), offset)
                branches.append(branch)
            tails = [sequences[i][common:] for i in selected]
            lengths = [len(tail) for tail in tails]
            width = max(lengths)
            inputs = mx.array([tail+[self.pad_id]*(width-len(tail)) for tail in tails])
            hidden = self.model.model(inputs, cache=branches)
            last = hidden[mx.arange(len(selected)), mx.array(lengths)-1].astype(mx.float32)
            probability = mx.softmax(last@self.answer_weight.T, axis=-1)[:, 1]
            mx.eval(probability)
            for index, value in zip(selected, probability.tolist()):
                scores[index] = float(value)
        return dict(scores=scores, common_prefix_tokens=common)
