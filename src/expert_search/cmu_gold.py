"""CMU Gold annotation adapter; missing annotations never become negative labels."""
import csv
import json
import math
from itertools import combinations
from pathlib import Path


def load_annotations(path):
    reviewers={}
    with Path(path).open() as f:
        for row in csv.DictReader(f,delimiter='\t'):
            grades={}
            for i in range(1,11):
                paper=row.get(f'Paper{i}','');score=row.get(f'Expertise{i}','')
                if paper and score and paper.lower()!='nan' and score.lower()!='nan':
                    value=float(score)
                    if not math.isfinite(value) or not 1<=value<=5:raise ValueError('Invalid expertise grade')
                    grades[paper]=value
            reviewers[row['ParticipantID']]=grades
    return reviewers


def pairwise_loss(predictions,annotations):
    loss=0.;weight=0.;pairs=0
    for reviewer,grades in annotations.items():
        if reviewer not in predictions or set(grades)-set(predictions[reviewer]):
            raise ValueError('Missing predictions for annotated pairs')
        for a,b in combinations(grades,2):
            delta=grades[a]-grades[b]
            if delta==0:continue
            prediction=predictions[reviewer][a]-predictions[reviewer][b]
            weight+=abs(delta);pairs+=1
            loss+=abs(delta)*(0 if prediction*delta>0 else .5 if prediction==0 else 1)
    return dict(weighted_pairwise_loss=loss/weight if weight else None,comparable_pairs=pairs)


def audit(path='artifacts/public/cmu-gold/data/evaluations.csv',output='docs/exploration/cmu-gold-audit.json'):
    rows=load_annotations(path)
    report=dict(reviewers=len(rows),annotations=sum(map(len,rows.values())),papers=len({p for r in rows.values() for p in r}),
        task='Within-reviewer ordering of annotated papers; not all-reviewer retrieval',
        annotation_license='CC BY 4.0; third-party metadata excluded; see upstream LICENSE-DATA.md')
    Path(output).write_text(json.dumps(report,indent=2));return report
