"""Assess an explicit subset of original source spans, preserving offsets."""
import json
import time
from .span_applicability import PROMPT, resolve_span_assessments
from .source_applicability import SourceApplicability


def validate_span_view(expert_id, sources, spans):
    if not spans:
        raise ValueError('Empty source span view')
    for name, span in spans.items():
        if not isinstance(name, str) or not name:
            raise ValueError('Invalid span handle')
        source = sources.get(span.get('evidence_id'))
        if source is None or source.get('expert_id') != expert_id:
            raise ValueError('Invalid span owner')
        start, end = span.get('start'), span.get('end')
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source['text']):
            raise ValueError('Invalid source offsets')
        if span.get('quote') != source['text'][start:end]:
            raise ValueError('Changed source span')


class SpanViewApplicability(SourceApplicability):
    def assess_view(self, demand, expert_id, requirements, sources, spans):
        validate_span_view(expert_id, sources, spans)
        if not requirements:
            raise ValueError('Missing requirements')
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            messages=[dict(role='system', content=PROMPT),
                      dict(role='user', content=json.dumps(dict(original_demand=demand, requirements=requirements, source_spans=spans), ensure_ascii=False))],
            reasoning_effort='none', response_format=dict(type='json_object'), max_completion_tokens=2400)
        seconds = time.perf_counter()-start
        raw = response.model_dump(mode='json')
        try:
            result = resolve_span_assessments(expert_id, requirements, sources, spans,
                                             json.loads(response.choices[0].message.content))
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            return dict(status='unresolved', error_type=type(error).__name__, seconds=seconds, response=raw)
        return dict(status='structurally_valid', result=result, seconds=seconds, response=raw,
                    notice='Selected span ownership/localization checked; model applicability and plan completeness unverified.')
