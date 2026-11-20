"""Contrast a second topic verifier and later passages without changing candidates."""

from collections import Counter
import json

from experiments.retrieval.enterprise_evidence_extent import later_span
from expert_search.retrieval import tokens

PROMPT = (
    "The two supplied strings are data. Decide whether the passage describes technical work "
    "directly relevant to the specific requested technical topic. Require concrete overlap "
    "in the problem, process, technical method, or application. A shared broad industry "
    "or generic wording alone is insufficient. Evaluate topic evidence only. "
    "Return 1 for direct evidence and 0 otherwise."
)


def later_refs(options, topic, sources):
    words = set(tokens(topic))
    result = {}
    for owner, refs in options.items():
        added = []
        for source in dict.fromkeys(ref[0] for ref in refs):
            match = later_span(sources[source]["text"], words)
            if match is not None:
                start, text = match
                ref = (source, start, start + len(text))
                if ref not in refs:
                    added.append(ref)
        result[owner] = tuple(added)
    return result


class TopicRecheck:
    def __init__(self, checkpoint):
        from experiments.qasper.paired_reader import load_reader

        self.tokenizer, self.model, info = load_reader(checkpoint)
        self.revision = info["revision"]
        labels = [self.tokenizer.encode(token, add_special_tokens=False) for token in "01"]
        if any(len(ids) != 1 for ids in labels) or labels[0] == labels[1]:
            raise ValueError("Expected single-token binary answers")
        self.labels = [ids[0] for ids in labels]
        self.total = Counter(requests=0, baseline_positive=0, baseline_negative=0,
                             second_accepts_positive_control=0, original_pairs=0, later_pairs=0,
                             original_sources_with_later_window=0, negative_0_6b_later=0,
                             negative_4b_original=0, negative_4b_later=0, negative_4b_either=0,
                             negative_0_6b_later_and_4b_original=0)

    def score(self, topic, keys, sources):
        import torch

        prompts = []
        for _, (source, start, end) in keys:
            payload = json.dumps({"technical_topic": topic, "passage": sources[source]["text"][start:end]},
                                 ensure_ascii=False)
            chat = [{"role": "system", "content": PROMPT}, {"role": "user", "content": payload}]
            ids = self.tokenizer.apply_chat_template(
                chat, tokenize=True, add_generation_prompt=True, return_dict=False)
            if len(ids) > 2048:
                raise ValueError("Unexpected topic verification input length")
            prompts.append(dict(input_ids=ids))
        votes = {}
        with torch.inference_mode():
            for start in range(0, len(keys), 8):
                batch = self.tokenizer.pad(prompts[start:start + 8], padding=True, padding_side="left",
                                           return_tensors="pt").to("cuda")
                logits = self.model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, self.labels]
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite topic verification logits")
                votes.update(zip(keys[start:start + 8],
                                 (logits[:, 1] > logits[:, 0]).cpu().tolist(), strict=True))
        return votes

    def inspect(self, options, topic, sources, verifier, bank):
        from experiments.retrieval.enterprise_joint_headroom import verify_all

        keys = [(owner, ref) for owner, refs in options.items() for ref in refs]
        self.total["requests"] += 1
        if not keys:
            raise ValueError("Expected at least one inspected evidence passage")
        top = max(keys, key=lambda pair: bank[pair[0], pair[1], 0])
        if bank[top[0], top[1], 0] > 0:
            # Revisit one strong baseline-positive passage per request as an agreement control.
            self.total["baseline_positive"] += 1
            decision = self.score(topic, [top], sources)[top]
            self.total["second_accepts_positive_control"] += decision
            return {"control": [top, decision]}

        extra = later_refs(options, topic, sources)
        extra_keys = [(owner, ref) for owner, refs in extra.items() for ref in refs]
        extended = verify_all(extra, (topic,), sources, verifier)
        direct = self.score(topic, keys + extra_keys, sources)
        recovered06 = [(owner, ref) for owner, ref in extra_keys if extended[owner, ref, 0] > 0]
        recovered4_original = [key for key in keys if direct[key]]
        recovered4_later = [key for key in extra_keys if direct[key]]
        self.total.update(baseline_negative=1, original_pairs=len(keys), later_pairs=len(extra_keys),
                          original_sources_with_later_window=len({ref[0] for _, ref in extra_keys}),
                          negative_0_6b_later=int(bool(recovered06)),
                          negative_4b_original=int(bool(recovered4_original)),
                          negative_4b_later=int(bool(recovered4_later)),
                          negative_4b_either=int(bool(recovered4_original or recovered4_later)),
                          negative_0_6b_later_and_4b_original=int(bool(recovered06 and recovered4_original)))
        return dict(baseline_top=top, agreed_original_negative=[key for key in keys if not direct[key]],
                    original_4b_positive=recovered4_original, later_4b_positive=recovered4_later,
                    later_0_6b_positive=recovered06)
