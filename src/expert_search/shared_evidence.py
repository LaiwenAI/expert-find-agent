"""Lossless text factoring with reversible, candidate-specific provenance."""


def factor_candidates(candidates):
    aliases={};sources=[];output=[]
    for candidate in candidates:
        references=[]
        for evidence in candidate['evidence']:
            if 'source_id' in evidence or not isinstance(evidence.get('text'),str):raise ValueError('Invalid inline evidence')
            text=evidence['text']
            if text not in aliases:
                aliases[text]=len(sources);sources.append(dict(source_id=aliases[text],text=text))
            references.append(dict({k:v for k,v in evidence.items() if k!='text'},source_id=aliases[text]))
        output.append(dict({k:v for k,v in candidate.items() if k!='evidence'},evidence=references))
    return dict(candidates=output,source_windows=sources)


def expand_candidates(shared):
    sources={}
    for source in shared['source_windows']:
        identity=source['source_id']
        if type(identity)!=int or identity in sources or not isinstance(source.get('text'),str):raise ValueError('Invalid shared source IDs')
        sources[identity]=source['text']
    output=[]
    for candidate in shared['candidates']:
        evidence=[]
        for reference in candidate['evidence']:
            identity=reference.get('source_id')
            if type(identity)!=int or identity not in sources:raise ValueError('Unknown source reference')
            evidence.append(dict({k:v for k,v in reference.items() if k!='source_id'},text=sources[identity]))
        output.append(dict({k:v for k,v in candidate.items() if k!='evidence'},evidence=evidence))
    return output
