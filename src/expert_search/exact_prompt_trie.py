"""Pack exact token prefixes into one preordered causal attention tree."""

from dataclasses import dataclass


def prefix_length(left, right):
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i
    return min(len(left), len(right))


@dataclass(frozen=True)
class PromptTree:
    tokens: tuple[int, ...]
    depth: tuple[int, ...]
    subtree_end: tuple[int, ...]
    leaves: tuple[int, ...]


def pack_prompts(prompts, *, common_only=False):
    """Return a forest whose valid attention edges are ancestor intervals.

    Node j is an ancestor of node i exactly when j <= i < subtree_end[j].
    Depth gives each node's original token position within its own prompt.
    """
    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError("Expected nonempty prompts")
    ranking = sorted(range(len(prompts)), key=prompts.__getitem__)
    root = prefix_length(prompts[ranking[0]], prompts[ranking[-1]])
    tokens, depth, ends, path = [], [], [], []
    leaves = [0] * len(prompts)
    previous = ()
    for original in ranking:
        prompt = prompts[original]
        shared = prefix_length(previous, prompt)
        if common_only:
            shared = min(shared, root)
        for node in path[shared:]:
            ends[node] = len(tokens)
        path = path[:shared]
        for index in range(shared, len(prompt)):
            path.append(len(tokens))
            tokens.append(prompt[index])
            depth.append(index)
            ends.append(0)
        leaves[original] = path[-1]
        previous = prompt
    for node in path:
        ends[node] = len(tokens)
    return PromptTree(tuple(tokens), tuple(depth), tuple(ends), tuple(leaves))
