"""Lossless bilingual alignment anchors; mechanical guards are not semantic gold."""
from collections import Counter
from decimal import Decimal
import re
import unicodedata

UNITS={'%':'%','percent':'%','nm':'nm','nanometers':'nm','nanometres':'nm','μm':'μm','micrometers':'μm','micrometres':'μm','mm':'mm','millimeters':'mm','millimetres':'mm','cm':'cm','centimeters':'cm','mM':'mM','millimolar':'mM','ms':'ms','milliseconds':'ms','ns':'ns','nanoseconds':'ns','MHz':'MHz','GHz':'GHz','kHz':'kHz','Hz':'Hz','MPa':'MPa','GPa':'GPa','kW':'kW','W':'W','kg':'kg','\u516c\u65a4':'kg','\u5343\u514b':'kg','\u6beb\u7c73':'mm','\u5fae\u7c73':'μm','\u7eb3\u7c73':'nm','\u6beb\u79d2':'ms','\u7eb3\u79d2':'ns'}
NUMBER=re.compile(r'(?<![A-Za-z0-9.])[-+−]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?')
UNIT=re.compile(r'\s*('+ '|'.join(re.escape(s) for s in sorted(UNITS,key=len,reverse=True))+r')(?![A-Za-z])')
OPERATORS=[('>=','(?:not less than|no less than|at least|\u4e0d\u4f4e\u4e8e|\u4e0d\u5c11\u4e8e|\u4e0d\u5c0f\u4e8e|\u5927\u4e8e\u7b49\u4e8e|\u81f3\u5c11|\u2265|>=)\\s*$'),('<=','(?:not more than|no more than|at most|up to|\u4e0d\u8d85\u8fc7|\u4e0d\u5927\u4e8e|\u5c0f\u4e8e\u7b49\u4e8e|\u81f3\u591a|\u2264|<=)\\s*$'),('>','(?:greater than|more than|over|above|\u5927\u4e8e|\u8d85\u8fc7|>)\\s*$'),('<','(?:less than|below|\u5c0f\u4e8e|\u4f4e\u4e8e|<)\\s*$'),('+/-',r'(?:plus or minus|±)\s*$')]


def lossless_segments(text,width=600):
    if not isinstance(text,str) or not text or type(width) is not int or width<1:raise ValueError('Invalid alignment input')
    output=[];start=0
    while start<len(text):
        end=min(start+width,len(text))
        if end<len(text):
            breaks=[m.end() for m in re.finditer('[\u3002\uff1b;!?\uff01\uff1f\\n]',text[start:end])]
            if breaks and breaks[-1]>=width//2:end=start+breaks[-1]
        output.append(dict(id='s'+str(len(output)),start=start,end=end,zh=text[start:end]));start=end
    assert ''.join(r['zh'] for r in output)==text
    return output


def numeric_anchors(text):
    normalized=[];positions=[]
    for index,char in enumerate(text):
        chunk=unicodedata.normalize('NFKC',char);normalized.append(chunk);positions.extend([index]*len(chunk))
    normalized=''.join(normalized);anchors=[]
    for match in NUMBER.finditer(normalized):
        literal=match.group();value=str(Decimal(literal.replace(',','').replace('−','-')).normalize());prefix=normalized[max(0,match.start()-40):match.start()];suffix=normalized[match.end():];unit_match=UNIT.match(suffix);unit=UNITS[unit_match.group(1)] if unit_match else None;operator=None
        for name,pattern in OPERATORS:
            if re.search(pattern,prefix,flags=re.IGNORECASE):operator=name;break
        post=suffix[unit_match.end():] if unit_match else suffix
        if re.match('\\s*\u4ee5\u4e0a',post):operator='>='
        elif re.match('\\s*\u4ee5\u4e0b',post):operator='<='
        anchors.append(dict(start=positions[match.start()],end=positions[match.end()-1]+1,literal=text[positions[match.start()]:positions[match.end()-1]+1],value=value,unit=unit,operator=operator))
    return anchors


def check_numeric_alignment(zh,en):
    source=numeric_anchors(zh);target=numeric_anchors(en)
    signature=lambda rows:[(r['value'],r['unit'],r['operator']) for r in rows]
    return dict(exact_ordered_supported_anchors_match=signature(source)==signature(target),source_anchors=source,target_anchors=target,scope='Limited Arabic-number/recognized-unit/comparator grammar, not complete semantic equivalence. Unparsed quantities, negation, object scope, dates, abbreviations and scientifically equivalent reformatting need separate audit; matching anchors never certifies translation.')
