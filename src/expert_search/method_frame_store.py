"""Bind cached, nonexhaustive method frames to exact owned source views."""
import hashlib
import json
from .span_view_applicability import validate_span_view


def bind_frames(source,record):
    if record['source_id']!=source['id'] or record['expert_id']!=source['expert_id'] or record['source_sha256']!=hashlib.sha256(source['text'].encode()).hexdigest():raise ValueError('Stale or foreign method-frame cache')
    frames=record['frames'];seen=set()
    if not isinstance(frames,list) or len(frames)>2:raise ValueError('Invalid cached frame budget')
    for frame in frames:
        if frame['id'] in seen or frame['source_id']!=source['id'] or frame['expert_id']!=source['expert_id']:raise ValueError('Duplicate or foreign method frame')
        seen.add(frame['id'])
        if record.get('audit',{}).get('labels',{}).get(frame['id'])!='supported_frame':raise ValueError('Frame not admitted by stored model audit')
        citations=frame['citations']
        if not isinstance(citations,list) or not 1<=len(citations)<=2:raise ValueError('Invalid cached citations')
        validate_span_view(source['expert_id'],{source['id']:source},{str(i):c for i,c in enumerate(citations)})
        for name in ['operation','input','output']:
            value=frame[name]
            if value is None and name!='operation':continue
            if not isinstance(value,str) or not value.strip() or len(value)>180:raise ValueError('Invalid cached field')
        conditions=frame['conditions']
        if not isinstance(conditions,list) or len(conditions)>2 or any(not isinstance(c,str) or not c.strip() or len(c)>180 for c in conditions):raise ValueError('Invalid cached conditions')
    return dict(source_id=source['id'],expert_id=source['expert_id'],source_sha256=record['source_sha256'],frames=frames,
                nonexhaustive=True,raw_source_required=True,scope='Recorded same-model screening only; source binding is not semantic certification. Null fields mean unreported, never impossible. Source remains eligible for raw retrieval regardless of frames.')


def as_fact_card(bound):
    """Adapter for existing fact-card consumers; preserves every populated field."""
    cards=[]
    for frame in bound['frames']:
        fields={k:frame[k] for k in ['input','operation','output','conditions']}
        claim='Source-only method frame; null means unreported, not nonexistent. '+json.dumps(fields,ensure_ascii=False)
        cards.append(dict(id=frame['id'],kind='method',claim=claim,citations=frame['citations']))
    return dict(source_id=bound['source_id'],expert_id=bound['expert_id'],source_sha256=bound['source_sha256'],cards=cards,nonexhaustive=True,notice=bound['scope'])
