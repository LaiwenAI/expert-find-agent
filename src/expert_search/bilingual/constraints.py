"""Lossless bilingual alignment anchors; mechanical guards are not semantic gold."""
from collections import Counter
from decimal import Decimal
import re
import unicodedata

UNITS={'%':'%','percent':'%','nm':'nm','nanometers':'nm','nanometres':'nm','μm':'μm','micrometers':'μm','micrometres':'μm','mm':'mm','millimeters':'mm','millimetres':'mm','cm':'cm','centimeters':'cm','mM':'mM','millimolar':'mM','ms':'ms','milliseconds':'ms','ns':'ns','nanoseconds':'ns','MHz':'MHz','GHz':'GHz','kHz':'kHz','Hz':'Hz','MPa':'MPa','GPa':'GPa','kW':'kW','W':'W','kg':'kg','公斤':'kg','千克':'kg','毫米':'mm','微米':'μm','纳米':'nm','毫秒':'ms','纳秒':'ns'}
NUMBER=re.compile(r'(?<![A-Za-z0-9.])[-+−]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?')
UNIT=re.compile(r'\s*('+ '|'.join(re.escape(s) for s in sorted(UNITS,key=len,reverse=True))+r')(?![A-Za-z])')
OPERATORS=[('>=',r'(?:not less than|no less than|at least|不低于|不少于|不小于|大于等于|至少|≥|>=)\s*$'),('<=',r'(?:not more than|no more than|at most|up to|不超过|不大于|小于等于|至多|≤|<=)\s*$'),('>',r'(?:greater than|more than|over|above|大于|超过|>)\s*$'),('<',r'(?:less than|below|小于|低于|<)\s*$'),('+/-',r'(?:plus or minus|±)\s*$')]


def lossless_segments(text,width=600):
    if not isinstance(text,str) or not text or type(width) is not int or width<1:raise ValueError('Invalid alignment input')
    output=[];start=0
    while start<len(text):
        end=min(start+width,len(text))
        if end<len(text):
            breaks=[m.end() for m in re.finditer(r'[。；;!?！？\n]',text[start:end])]
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
        if re.match(r'\s*以上',post):operator='>='
        elif re.match(r'\s*以下',post):operator='<='
        anchors.append(dict(start=positions[match.start()],end=positions[match.end()-1]+1,literal=text[positions[match.start()]:positions[match.end()-1]+1],value=value,unit=unit,operator=operator))
    return anchors


def check_numeric_alignment(zh,en):
    source=numeric_anchors(zh);target=numeric_anchors(en)
    signature=lambda rows:[(r['value'],r['unit'],r['operator']) for r in rows]
    return dict(exact_ordered_supported_anchors_match=signature(source)==signature(target),source_anchors=source,target_anchors=target,scope='Limited Arabic-number/recognized-unit/comparator grammar, not complete semantic equivalence. Unparsed quantities, negation, object scope, dates, abbreviations and scientifically equivalent reformatting need separate audit; matching anchors never certifies translation.')
