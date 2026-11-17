"""Non-numeric controlled contrasts for a grounding judge; expected labels stay outside requests."""
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--paraphrase', action='store_true')
    parser.add_argument('--cross-language', action='store_true')
    args = parser.parse_args()
    if args.paraphrase and args.cross_language:
        parser.error('Choose one cohort')
    root = Path('artifacts/grounding-nonnumeric-cross-language-controls-v1' if args.cross_language else 'artifacts/grounding-nonnumeric-paraphrase-controls-v1' if args.paraphrase else 'artifacts/grounding-nonnumeric-scope-controls-v1')
    root.mkdir(parents=True, exist_ok=False)
    selected = cases(); tasks = []
    if args.paraphrase:
        rewrites = [
            'Seek a chromium-free coating.',
            'Seek a non-destructive technique to examine the interior of pipes.',
            'Seek a recyclable adhesive for bonding glass to aluminum.',
            'Seek a process capable of operating in wet environments.',
            'Seek a catalyst to desulfurize diesel rather than gasoline.',
            'Seek a way to reduce vibration while retaining the existing motor.',
            'Seek an expert able to investigate seawater corrosion.',
            'Seek an approach that could lower energy use during drying.',
        ]
        for i, rewrite in enumerate(rewrites):
            selected[2 * i]['derived'] = rewrite
    if args.cross_language:
        translations = [
            ('\u5bfb\u627e\u4e0d\u542b\u94ec\u7684\u6d82\u5c42\u3002', '\u5bfb\u627e\u542b\u94ec\u7684\u6d82\u5c42\u3002'),
            ('\u5bfb\u627e\u65e0\u9700\u5207\u5272\u7ba1\u9053\u5373\u53ef\u68c0\u67e5\u5176\u5185\u90e8\u7684\u65b9\u6cd5\u3002', '\u5bfb\u627e\u65e0\u9700\u5207\u5272\u7ba1\u9053\u5373\u53ef\u68c0\u67e5\u5176\u5916\u90e8\u7684\u65b9\u6cd5\u3002'),
            ('\u5bfb\u627e\u80fd\u7c98\u63a5\u94dd\u548c\u73bb\u7483\u7684\u53ef\u56de\u6536\u7c98\u5408\u5242\u3002', '\u5bfb\u627e\u80fd\u7c98\u63a5\u94dd\u548c\u94a2\u7684\u53ef\u56de\u6536\u7c98\u5408\u5242\u3002'),
            ('\u5bfb\u627e\u80fd\u591f\u5728\u6f6e\u6e7f\u6761\u4ef6\u4e0b\u5de5\u4f5c\u7684\u5de5\u827a\u3002', '\u5bfb\u627e\u53ea\u80fd\u5728\u5e72\u71e5\u6761\u4ef6\u4e0b\u5de5\u4f5c\u7684\u5de5\u827a\u3002'),
            ('\u5bfb\u627e\u4ece\u67f4\u6cb9\u800c\u975e\u6c7d\u6cb9\u4e2d\u53bb\u9664\u786b\u7684\u50ac\u5316\u5242\u3002', '\u5bfb\u627e\u4ece\u6c7d\u6cb9\u800c\u975e\u67f4\u6cb9\u4e2d\u53bb\u9664\u786b\u7684\u50ac\u5316\u5242\u3002'),
            ('\u5bfb\u627e\u4e0d\u66f4\u6362\u7535\u673a\u5373\u53ef\u964d\u4f4e\u632f\u52a8\u7684\u65b9\u6cd5\u3002', '\u5bfb\u627e\u901a\u8fc7\u66f4\u6362\u7535\u673a\u6765\u964d\u4f4e\u632f\u52a8\u7684\u65b9\u6cd5\u3002'),
            ('\u5bfb\u627e\u80fd\u591f\u7814\u7a76\u6d77\u6c34\u8150\u8680\u7684\u4e13\u5bb6\u3002', '\u5bfb\u627e\u5df2\u7ecf\u6d88\u9664\u6d77\u6c34\u8150\u8680\u7684\u4e13\u5bb6\u3002'),
            ('\u5bfb\u627e\u53ef\u80fd\u964d\u4f4e\u5e72\u71e5\u80fd\u8017\u7684\u65b9\u6cd5\u3002', '\u5bfb\u627e\u4fdd\u8bc1\u964d\u4f4e\u5e72\u71e5\u80fd\u8017\u7684\u65b9\u6cd5\u3002'),
        ]
        for i, (faithful, altered) in enumerate(translations):
            selected[2 * i]['derived'] = faithful
            selected[2 * i + 1]['derived'] = altered
    for case in selected:
        for order in [(0, 1, 2), (2, 1, 0)]:
            identity = dict(case_id=case['id'], order=order, payload={'source':case['source'], 'derived':case['derived']},
                            questions=[dict(type='choice', name='grounding', instructions=GROUNDING_INSTRUCTION, choices=[CHOICES[i] for i in order])])
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            tasks.append((case, identity, key))
    protocol = dict(model='gpt-6-luna', cases=selected, tasks=len(tasks), choices=CHOICES, instruction=GROUNDING_INSTRUCTION,
                    scope='Handwritten non-numeric synthetic controls; faithful examples are paraphrases with --paraphrase or Chinese translations with --cross-language. Expected labels are not sent to the model; template pairs are dependent. This does not measure real-data accuracy.')
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
