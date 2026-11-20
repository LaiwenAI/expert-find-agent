"""Keep unverified request text attached to a structured role interpretation."""

from dataclasses import dataclass
import re


_BOUNDARY = re.compile(r"(?<=[.!?;])\s+|(?<=[\u3002\uff01\uff1f\uff1b])\s*|\r?\n+")
_OPTIONAL_PREFIX = re.compile(
    r"^(?:optional(?:\s+condition)?\s*:|(?:this|the) additional condition is optional\s*:)",
    re.IGNORECASE,
)
_MANDATORY_CUE = re.compile(r"\b(?:must|shall|required|mandatory|need to|prerequisite)\b|(?:\u5fc5\u987b|\u987b\u5177\u5907|\u5fc5\u9700|\u4e0d\u5f97|\u5e94\u5f53)")


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def quote(self, text: str) -> str:
        return text[self.start:self.end]


@dataclass(frozen=True)
class RequirementPacket:
    role: Span | None
    optional: tuple[Span, ...]
    unresolved: tuple[Span, ...]

    @property
    def role_only(self) -> bool:
        return self.role is not None and not self.unresolved


def clauses(text: str) -> tuple[Span, ...]:
    """Return trimmed source intervals, retaining punctuation in the left clause."""
    result = []
    start = 0
    for boundary in _BOUNDARY.finditer(text):
        _append_span(result, text, start, boundary.start())
        start = boundary.end()
    _append_span(result, text, start, len(text))
    return tuple(result)


def _append_span(output: list[Span], text: str, start: int, end: int) -> None:
    content = text[start:end]
    left = start + len(content) - len(content.lstrip())
    right = start + len(content.rstrip())
    if left < right:
        output.append(Span(left, right))


def form_packet(text: str, interpreted_role: Span | None) -> RequirementPacket:
    """Cancel only explicit whole-clause preferences; leave every other clause open.

    The role span is a trusted output of a separate role-only interpretation.
    Matching a raw clause by offset does not prove that interpretation.
    """
    inventory = clauses(text)
    if interpreted_role is not None and interpreted_role not in inventory:
        raise ValueError("A role interpretation must identify one whole verbatim clause")

    optional, unresolved = [], []
    for span in inventory:
        if span == interpreted_role:
            continue
        quote = span.quote(text)
        marker = _OPTIONAL_PREFIX.match(quote)
        if marker and not _MANDATORY_CUE.search(quote[marker.end():].casefold()):
            optional.append(span)
        else:
            unresolved.append(span)
    return RequirementPacket(interpreted_role, tuple(optional), tuple(unresolved))
