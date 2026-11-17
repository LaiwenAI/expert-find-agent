"""Keep report prose English and escape original-language source data."""
from pathlib import Path
import re

_SOURCE_CHARACTERS=re.compile(r'[\u2e80-\u2fdf\u3000-\u303f\u3100-\u312f\u31a0-\u31bf\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef\U00020000-\U0003134f]')


def escape_source_text(text):
    """Escape ideographs/fullwidth punctuation, preserving English typography.

    Human-authored report labels must already be English. Original UTF-8 data
    stays in local artifacts; this display representation is not a translation.
    """
    if not isinstance(text,str):raise TypeError('Report text must be a string')
    return _SOURCE_CHARACTERS.sub(lambda match: ('\\u'+format(ord(match.group()),'04x')) if ord(match.group())<=0xffff else ('\\U'+format(ord(match.group()),'08x')),text)


def write_english_report(path,text):
    target=Path(path)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(escape_source_text(text),encoding='utf-8')
