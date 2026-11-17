"""Flag a limited set of spelled-out quantities beyond the v2 numeric grammar."""
import re
import unicodedata
from .postvalidation import check_pair

# Conservative triggers, deliberately not a Chinese or English number parser.
ZH_DIGITS=r'[\u96f6\u3007\u4e00\u4e8c\u4e24\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343]+'
ZH_UNITS=r'(?:\u5e74|\u6708|\u65e5|\u5929|\u5c0f\u65f6|\u5206\u949f|\u79d2|\u6b21|\u4e2a|\u53f0|\u4ef6|\u516c\u65a4|\u5343\u514b|\u6beb\u7c73|\u5fae\u7c73|\u7eb3\u7c73|\u5143|%)'
ZH_PATTERN=re.compile(r'(?:'+ZH_DIGITS+ZH_UNITS+r'|[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u4e24\u5341\u767e\u53430-9]+[\u4e07\u4ebf]|\u767e\u5206\u4e4b'+ZH_DIGITS+r')')
EN_NUMBER=r'(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)'
EN_UNIT=r'(?:years?|months?|days?|hours?|minutes?|seconds?|times?|items?|units?|percent|kilograms?|millimet(?:er|re)s?|micromet(?:er|re)s?|nanomet(?:er|re)s?|watts?|volts?)'
EN_PATTERN=re.compile(r'\b(?:'+EN_NUMBER+r'(?:[\s-]+'+EN_NUMBER+r')*\s+'+EN_UNIT+r'|(?:\d+(?:\.\d+)?|a|'+EN_NUMBER+r')\s+(?:hundred|thousand|million|billion))\b',re.I)


def audit(zh,en):
    """Downgrade v2 to uncertain on detected unsupported expression; never promote."""
    base=check_pair(zh,en)
    zh_hit=bool(ZH_PATTERN.search(unicodedata.normalize('NFKC',zh)))
    en_hit=bool(EN_PATTERN.search(unicodedata.normalize('NFKC',en)))
    return {'version':'numeric-word-audit-v1','status':'uncertain' if zh_hit or en_hit else base['status'],
            'v2_status':base['status'],'source_word_quantity_flag':zh_hit,'target_word_quantity_flag':en_hit,
            'scope':'Partial word-quantity detection only; flags can be nonquantitative and many forms are missed. No numeric normalization, semantic equivalence, correction, or relevance claim.'}
