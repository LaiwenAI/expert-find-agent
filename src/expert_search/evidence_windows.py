"""Select source-ordered lexical windows at native sentence boundaries."""

from .retrieval import tokens


DOC_SENTENCES = 3


def choose_windows(claim, sentences):
    wanted = set(tokens(claim))
    sentence_terms = [set(tokens(sent)) & wanted for sent in sentences]
    width = min(DOC_SENTENCES, len(sentences))
    windows = [tuple(range(i, i + width)) for i in range(len(sentences) - width + 1)]
    words = [set().union(*(sentence_terms[i] for i in block)) for block in windows]
    first = max(range(len(windows)), key=lambda i: (len(words[i]), -i))
    allowed = [i for i, block in enumerate(windows)
               if not set(block).intersection(windows[first]) and words[i] - words[first]]
    if not allowed:
        return windows[first], windows[first]
    second = max(allowed, key=lambda i: (len(words[i] - words[first]), -i))
    return windows[first], tuple(sorted((*windows[first], *windows[second])))
