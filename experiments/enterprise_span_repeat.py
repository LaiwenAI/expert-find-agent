"""One matched repeat to distinguish context effects from judge variability."""
import hashlib
import json
from pathlib import Path
import enterprise_span_assessment as experiment


def run():
    parent=Path('artifacts/enterprise-span-assessment-v1')
    root=Path('artifacts/enterprise-span-repeat-v1');root.mkdir(exist_ok=True)
    protocol=dict(purpose='One fresh same-payload repeat after context disagreements. No retries, same model/prompt/selection/budgets. Identical payload sets verified after run. Diagnose variability, not independent gold or latency replication.',
                  parent_protocol_sha256=hashlib.sha256((parent/'protocol.json').read_bytes()).hexdigest(),
                  parent_response_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((parent/'responses').glob('*.json'))},
                  wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    marker=root/'repeat-parent.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed repeated-assessment parent')
    marker.write_text(json.dumps(protocol,indent=2))
    experiment.ROOT=root
    experiment.run()
    if json.loads((parent/'protocol.json').read_text())!=json.loads((root/'protocol.json').read_text()):raise ValueError('Repeat payload protocol differs')
    results=[]
    for path in sorted((parent/'responses').glob('*.json')):
        first=json.loads(path.read_text());second=json.loads((root/'responses'/path.name).read_text())
        result=dict(request=path.stem,first_status=first['status'],second_status=second['status'])
        if first['status']==second['status']=='structurally_valid':
            a=first['result']['assessments'];b=second['result']['assessments']
            if set(a)!=set(b):raise ValueError('Repeated requirements differ')
            actionable=lambda r:r['level'] in ('direct_application','transferable_method')
            result.update(requirements=len(a),level_agreement=sum(a[k]['level']==b[k]['level'] for k in a),actionable_agreement=sum(actionable(a[k])==actionable(b[k]) for k in a))
        results.append(result)
    valid=[r for r in results if 'requirements' in r]
    summary=dict(unique_requests=len(results),both_valid=len(valid),requirements=sum(r['requirements'] for r in valid),level_agreement=sum(r['level_agreement'] for r in valid),actionable_agreement=sum(r['actionable_agreement'] for r in valid),
                 warning='Unique input requests, not independent queries; same expert and demand repeated across contexts. Fresh same-model variability diagnostic, not correctness, probability calibration, statistical equivalence or paired latency.')
    (root/'repeat-analysis.json').write_text(json.dumps(dict(summary=summary,rows=results),indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env');run()
