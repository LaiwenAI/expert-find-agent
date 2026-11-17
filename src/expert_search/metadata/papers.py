"""Publication-time and citation metadata without inferring personal contribution."""
import calendar
from datetime import date,timedelta
import re


def publication_interval(raw):
    result=dict(raw=raw,precision=None,start=None,end_exclusive=None,status='missing' if raw is None or raw=='' else 'unparsed')
    if not isinstance(raw,str):return result
    m=re.fullmatch(r'(\d{4})(?:[/-](\d{1,2})(?:[/-](\d{1,2}))?)?',raw.strip())
    if not m:return result
    try:
        year=int(m[1]);month=int(m[2]) if m[2] else 1;day=int(m[3]) if m[3] else 1;start=date(year,month,day)
        if m[3]:end=start+timedelta(days=1);precision='day'
        elif m[2]:end=date(year+1,1,1) if month==12 else date(year,month+1,1);precision='month'
        else:end=date(year+1,1,1);precision='year'
    except (ValueError,OverflowError):result['status']='invalid_calendar_date';return result
    result.update(precision=precision,start=start.isoformat(),end_exclusive=end.isoformat(),status='parsed');return result


def availability(interval,as_of):
    cutoff=date.fromisoformat(as_of)
    if interval['status']!='parsed':return 'unknown_publication_date'
    if date.fromisoformat(interval['start'])>cutoff:return 'after_cutoff'
    if date.fromisoformat(interval['end_exclusive'])<=cutoff+timedelta(days=1):return 'before_or_on_cutoff'
    return 'overlaps_cutoff_date_precision_uncertain'


def count_value(value):
    if type(value) is int and value>=0:return value
    if isinstance(value,str) and re.fullmatch(r'\d+',value.strip()):return int(value.strip())
    return None


def citation_metadata(row):
    total=count_value(row.get('gs_scholarly_total_citations'));history=[];bad=0
    raw=row.get('gs_scholarly_citations_by_year')
    if isinstance(raw,list):
        for entry in raw:
            if not isinstance(entry,dict):bad+=1;continue
            year=count_value(entry.get('year'));n=count_value(entry.get('citations'))
            if year is None or not 1000<=year<=9999 or n is None:bad+=1;continue
            history.append(dict(year=year,citations=n))
    elif raw is not None:bad+=1
    duplicate_years=len({r['year'] for r in history})!=len(history)
    yearly_sum=sum(r['citations'] for r in history)
    status='unknown_total' if total is None else 'yearly_sum_exceeds_total' if yearly_sum>total else 'yearly_sum_matches_total' if history and yearly_sum==total else 'partial_or_unresolved_history'
    return dict(observed_total=total,by_year=history,invalid_history_entries=bad,duplicate_years=duplicate_years,total_history_status=status,observation_date=None,
                self_citations=None,field_normalized_impact=None,personal_credit=None,scope='As-provided paper-level counts, not personal contribution. Observation date/field/self-citations unknown; do not use totals for retrospective cutoffs or compare fields as normalized impact.')


def paper_metadata(row):
    return dict(publication=publication_interval(row.get('gs_publication_date')),citations=citation_metadata(row),doi=row.get('doi'),author_list_text=row.get('gs_authors'),
                authorship_association='provided_teacher_id_not_independently_verified',contribution=dict(status='no_explicit_role_in_source_fields',verified_roles=[],personal_method_credit=None,
                scope='Neither author order nor citation count establishes honorary authorship or a specific individual contribution. Paper association is team-level evidence until role/identity verification.'))
