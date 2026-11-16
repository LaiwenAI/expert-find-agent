"""Training-free English source-window packing, an experimental IR component."""
import math
import re
from collections import Counter

STOP=set('a an the and or of to in on for from by with as is are was were be been this that these those we our it its their using use used'.split())
PATTERN=re.compile(r'[A-Za-z][A-Za-z0-9-]*')


def terms(text):
    return {m.group().lower() for m in PATTERN.finditer(text) if m.group().lower() not in STOP}


def idf_weights(texts):
    texts=list(texts);df=Counter(t for text in texts for t in terms(text))
    return {t:math.log((len(texts)+1)/(n+1))+1 for t,n in df.items()}


def windows(document,query_terms,weights,max_chars):
    text=document['text'];maximum=max(0,len(text)-max_chars)
    starts={0}
    hits=[(weights.get(m.group().lower(),1.),m.start()) for m in PATTERN.finditer(text)
          if m.group().lower() in query_terms]
    for _,position in sorted(hits,key=lambda x:(-x[0],x[1]))[:5]:
        starts.add(min(maximum,max(0,position-max_chars//3)))
    return [dict(paper_id=document['id'],start=start,end=min(len(text),start+max_chars),
                 text=text[start:start+max_chars]) for start in sorted(starts) if text[start:start+max_chars]]


def pack(documents,query,weights,budget=2400,window_size=1200,mode='coverage'):
    if mode not in ('coverage','similarity') or budget<1 or not 1<=window_size<=budget:
        raise ValueError('Invalid packing parameters')
    if len({d['id'] for d in documents})!=len(documents):raise ValueError('Duplicate source IDs')
    if any(not isinstance(d['text'],str) or not math.isfinite(d['similarity']) for d in documents):
        raise ValueError('Invalid source text or similarity')
    if any(not math.isfinite(w) or w<=0 for w in weights.values()):raise ValueError('Invalid IDF weights')
    wanted=terms(query);total=sum(weights.get(t,1.) for t in sorted(wanted)) or 1.
    selected=[];covered=set();used=set();remaining=budget
    candidates=[]
    for document in sorted(documents,key=lambda d:d['id']):
        for window in windows(document,wanted,weights,window_size):
            candidates.append((document,window,terms(window['text'])&wanted))
    while remaining>0:
        options=[]
        for document,window,overlap in candidates:
            if document['id'] in used or len(window['text'])>remaining:continue
            gain=sum(weights.get(t,1.) for t in sorted(overlap-covered))/total
            semantic=max(0.,document['similarity'])
            if mode=='coverage':priority=(gain+.15*semantic)/math.sqrt(max(300,len(window['text'])))
            else:priority=semantic
            # The common lexical window tie-break keeps the comparison about
            # document selection, rather than arbitrary leading text alone.
            options.append((priority,sum(weights.get(t,1.) for t in sorted(overlap)),document['id'],window['start'],document,window,overlap))
        if not options:break
        best=min(options,key=lambda item:(-item[0],-item[1],item[2],item[3]))
        document,window,overlap=best[4:]
        selected.append(window);used.add(document['id']);covered.update(overlap);remaining-=len(window['text'])
    return sorted(selected,key=lambda w:(w['paper_id'],w['start']))
