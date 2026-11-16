"""Applicability with deterministic, owned source-span handles."""
import json
import time

from .applicability import assess_candidate
from .source_applicability import SourceApplicability, PROMPT as QUOTE_PROMPT

PROMPT = QUOTE_PROMPT.replace('"citations":[{"evidence_id":str,"quote":str}]', '"span_ids":[str]').replace(
    'Every non-insufficient judgment requires literal quotes from owned sources. Do not invent citations or paraphrase quotes.',
    'Every non-insufficient judgment requires supplied source span_ids supporting that judgment. Select IDs only; do not copy or rewrite quotes. Never invent span IDs.')


def source_spans(expert_id, sources, width=400):
    if type(width) is not int or width < 1:
        raise ValueError('Invalid span width')
    spans = {}
    for sid, source in sources.items():
        if source.get('expert_id') != expert_id:
            raise ValueError('Invalid evidence owner')
        for start in range(0, len(source['text']), width):
            end = min(start+width, len(source['text']))
            if source['text'][start:end].strip():
                spans['p'+str(len(spans))] = dict(evidence_id=sid, start=start, end=end,
                                                quote=source['text'][start:end])
    return spans


def resolve_span_assessments(expert_id, requirements, sources, spans, response):
    rows = response.get('assessments')
    if not isinstance(rows, list) or len(rows) != len(requirements):
        raise ValueError('Incomplete applicability assessments')
    assessments = {}
    for row in rows:
        rid = row.get('requirement_id')
        if rid not in requirements or rid in assessments:
            raise ValueError('Unknown or duplicate requirement')
        ids = row.get('span_ids')
        if not isinstance(ids, list) or any(not isinstance(i, str) or i not in spans for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('Invalid source span IDs')
        assessments[rid] = dict(level=row.get('level'), citations=[dict(spans[i]) for i in ids],
                                transfer_reason=row.get('transfer_reason'), assumptions=row.get('assumptions', []))
    return assess_candidate(expert_id, requirements, assessments, sources)


class SpanApplicability(SourceApplicability):
    def assess(self, demand, expert_id, requirements, sources):
        if not requirements or not sources:
            raise ValueError('Missing requirements or sources')
        spans = source_spans(expert_id, sources)
        payload = dict(original_demand=demand, requirements=requirements, source_spans=spans)
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            messages=[dict(role='system', content=PROMPT),
                      dict(role='user', content=json.dumps(payload, ensure_ascii=False))],
            reasoning_effort='none', response_format=dict(type='json_object'), max_completion_tokens=2400)
        seconds = time.perf_counter()-start
        raw = response.model_dump(mode='json')
        try:
            result = resolve_span_assessments(expert_id, requirements, sources, spans,
                                             json.loads(response.choices[0].message.content))
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            return dict(status='unresolved', error_type=type(error).__name__, seconds=seconds, response=raw)
        return dict(status='structurally_valid', result=result, seconds=seconds, response=raw,
                    notice='Deterministic span selection checks localization only, not semantic support or plausible transfer.')
