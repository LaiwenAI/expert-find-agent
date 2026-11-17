"""Inference-only MERIT released suitability benchmark input adapter.

Human ratings are deliberately excluded from inference cases. Reviewer suitability
is an auxiliary task, not evidence of industrial-project ability.
"""
import ast
import hashlib
import json
import re


def prompts(path):
    values={}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            if node.targets[0].id in ('action_system_prompt','action_user_prompt'):values[node.targets[0].id]=ast.literal_eval(node.value)
    if set(values)!={'action_system_prompt','action_user_prompt'}:raise ValueError('Missing official prompt strings')
    return values


def candidate_history(papers):
    if not isinstance(papers,list):raise ValueError('Invalid reviewer publication list')
    blocks=[]
    for i,paper in enumerate(papers,1):
        if not isinstance(paper,dict):raise ValueError('Invalid reviewer paper')
        title=paper.get('title') or '';abstract=paper.get('abstract') or ''
        if not isinstance(title,str) or not isinstance(abstract,str):raise ValueError('Invalid reviewer text')
        content=f'{i}. {title}'.strip()
        if abstract:content+='\n'+abstract
        if content:blocks.append(content)
    return '\n\n'.join(blocks) if blocks else 'None'


def build_case(row,index,official):
    paper=row['paper'];author=row['author']
    fields={k:paper.get(k) or '' for k in ('title','abstract','introduction')}
    if any(not isinstance(v,str) for v in fields.values()):raise ValueError('Invalid target paper text')
    history=candidate_history(author.get('papers',[]))
    user=official['action_user_prompt'].replace('{{paper_title}}',fields['title']).replace('{{paper_abstract}}','Abstract:\n'+fields['abstract']).replace('{{paper_introduction}}','Introduction:\n'+fields['introduction']).replace('{{candidate_history_list}}',history)
    return dict(id=f'pair-{index:04d}',input=user,input_sha256=hashlib.sha256(user.encode()).hexdigest(),paper_cluster=hashlib.sha256(' '.join(fields['title'].casefold().split()).encode()).hexdigest())


def gold(row):
    value=row.get('score')
    if type(value) not in (str,int,float):raise ValueError('Invalid self-rating')
    score=float(value)
    if score not in (1.,2.,3.,4.,5.):raise ValueError('Out-of-range self-rating')
    return int(score>3)


def parse_final_label(text):
    if not isinstance(text,str):raise ValueError('Missing generated text')
    # Require an explicit final field; do not read numbered requirements as labels.
    matches=re.findall(r'\[FINAL_LABEL\]\s*([01])\s*(?=$|\n|<)',text)
    if len(matches)!=1:raise ValueError('Missing or ambiguous final label')
    return int(matches[0])
