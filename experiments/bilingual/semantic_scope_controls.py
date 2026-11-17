"""Non-numeric controlled contrasts for a grounding judge; expected labels stay outside requests."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import json
from pathlib import Path
from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.choice_judgments import decide
from .audit_disagreement import CHOICES
from .grounding_control_cases import GROUNDING_INSTRUCTION


def cases():
    templates = [
        ('Find a coating that does not contain chromium.', 'Find a coating that contains chromium.'),
        ('Find a method for inspecting the inside of pipes without cutting them.', 'Find a method for inspecting the outside of pipes without cutting them.'),
        ('Find a recyclable adhesive that bonds aluminum to glass.', 'Find a recyclable adhesive that bonds aluminum to steel.'),
        ('Find a process that can work in wet conditions.', 'Find a process that only works in dry conditions.'),
        ('Find a catalyst for removing sulfur from diesel, not from gasoline.', 'Find a catalyst for removing sulfur from gasoline, not from diesel.'),
        ('Find a way to reduce vibration without replacing the motor.', 'Find a way to reduce vibration by replacing the motor.'),
        ('Find an expert who can study corrosion in seawater.', 'Find an expert who has already eliminated corrosion in seawater.'),
        ('Find an approach that may reduce energy consumption during drying.', 'Find an approach guaranteed to reduce energy consumption during drying.'),
    ]
    return [dict(id=f'scope-{i}-{kind}', source=source, derived=derived, expected=expected)
            for i, (source, changed) in enumerate(templates)
            for kind, derived, expected in [('faithful', source, 'supported'), ('altered', changed, 'changed_or_unsupported')]]


def main():
    root = Path('artifacts/grounding-nonnumeric-scope-controls-v1')
    root.mkdir(parents=True, exist_ok=False)
    selected = cases(); tasks = []
    for case in selected:
        for order in [(0, 1, 2), (2, 1, 0)]:
            identity = dict(case_id=case['id'], order=order, payload={'source':case['source'], 'derived':case['derived']},
                            questions=[dict(type='choice', name='grounding', instructions=GROUNDING_INSTRUCTION, choices=[CHOICES[i] for i in order])])
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            tasks.append((case, identity, key))
    protocol = dict(model='gpt-6-luna', cases=selected, tasks=len(tasks), choices=CHOICES, instruction=GROUNDING_INSTRUCTION,
                    scope='Handwritten non-numeric synthetic controls. Expected labels are not sent to the model; template pairs are dependent. This does not measure real-data accuracy.')
    (root/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (root/'predictions').mkdir()
    load_dotenv('.env')
    def execute(task):
        case, identity, key = task
        row = dict(key=key, case_id=case['id'], order=identity['order'], expected=case['expected'])
        try:
            result = decide(LaiwenAI(model='gpt-6-luna', max_retries=0), identity['payload'], identity['questions'])
            row.update(status='valid', prediction=result['labels']['grounding'], seconds=result['seconds'])
        except Exception as exc:
            row.update(status='failed', error_type=type(exc).__name__)
        (root/'predictions'/f'{key}.json').write_text(json.dumps(row, indent=2)+'\n')
        return row
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(execute, tasks))
    counts = Counter((r['expected'], r.get('prediction', r['status'])) for r in rows)
    summary = dict(tasks=len(rows), valid=sum(r['status']=='valid' for r in rows),
                   confusion=[dict(expected=a, predicted=b, count=n) for (a,b),n in sorted(counts.items())],
                   scope=protocol['scope'])
    (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
