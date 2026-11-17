"""Reject ideographs and fullwidth/CJK punctuation in Git-visible text/paths.

Raw data, local artifacts and environments follow .gitignore exclusions. Escaped
language fixtures remain usable at runtime. Git history is not rewritten.
"""
from pathlib import Path
import json
import re
import subprocess

PATTERN=re.compile(r'[\u2e80-\u2fdf\u3000-\u303f\u3100-\u312f\u31a0-\u31bf\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef\U00020000-\U0003134f]')


def check():
    root=Path(__file__).resolve().parents[2]
    try:
        paths=subprocess.check_output(['rg','--files','--hidden','-g','!.git/**','-g','!.env'],cwd=root,text=True).splitlines()
    except FileNotFoundError:
        paths=[p for p in subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=root).decode().split('\0') if p and p!='.env']
    violations=[];text_files=0
    for name in paths:
        if PATTERN.search(name):violations.append(dict(path=name,kind='path'))
        try:body=(root/name).read_text()
        except (UnicodeDecodeError,OSError):continue
        text_files+=1
        lines=[i for i,line in enumerate(body.splitlines(),1) if PATTERN.search(line)]
        if lines:violations.append(dict(path=name,kind='content',lines=lines))
    report=dict(text_files=text_files,violations=violations,scope='Current Git-visible working tree; ignored original data/artifacts and Git history excluded. Language fixtures may be Unicode-escaped.')
    print(json.dumps(report,indent=2))
    return not violations


if __name__=='__main__':raise SystemExit(0 if check() else 1)
