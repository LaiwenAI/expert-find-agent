"""Build a reviewable source archive, excluding credentials and third-party corpora."""
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path


def build(output):
    root=Path(__file__).resolve().parents[1]
    names=['experiments/README.md','docs/paper/sigir/README.md','README.md','LICENSE','pyproject.toml','requirements.txt','.env.example','.gitignore']
    for directory in ('src','tests','experiments'):
        names.extend(str(p.relative_to(root)) for p in (root/directory).rglob('*.py'))
    names.extend(str(p.relative_to(root)) for p in (root/'experiments').rglob('*.sbatch'))
    names.extend(str(p.relative_to(root)) for p in (root/'docs/assets').glob('*.svg'))
    names.extend(str(p.relative_to(root)) for p in (root/'benchmarks').rglob('*') if p.is_file() and p.suffix in ('.jsonl','.json','.md'))
    names.extend(['docs/grounding-review-protocol.md', 'docs/exploration/channel-fusion-20261117.md', 'docs/exploration/bilingual-audit-order-controls-20261117.md', 'docs/exploration/enterprise-review-readiness-20261117.md', 'docs/exploration/fixed-evidence-routing-pilot.md', 'docs/exploration/oversearch-cost-baseline.md'])
    names.extend(str(p.relative_to(root)) for p in (root/'docs/paper/sigir').rglob('*.tex'))
    names.extend(str(p.relative_to(root)) for p in (root/'docs/paper/sigir').rglob('*.json'))
    names.extend('docs/paper/sigir/template/'+n for n in ('acmart.cls','acmart.dtx','acmart.ins','ACM-Reference-Format.bst','README'))
    names.extend(['docs/scifact-evidence-verdict.md','docs/repository-language.md','docs/exploration/factored-decision-20261117.md','docs/exploration/bilingual-retrieval-20261117.md','docs/exploration/translation-pilot-20261117.md','docs/bilingual-query-data-card.md','docs/merit-suitability-card.md','docs/fresh-evaluation-data-card.md','docs/transfer-cases-card.md','docs/evidencebench-card.md','docs/exploration/topconf-review-20261117.md','docs/research-methods.md','docs/benchmark-results.md','docs/annotation-protocol.md','docs/exploration/pageindex-expert-design.md','docs/exploration/closest-work-review.md','docs/exploration/reproduction-map.md','docs/cluster.md','docs/paper-controls-data-card.md','docs/paper/references.bib','docs/usage.md','docs/open-source.md','docs/scoped-dataset-card.md'])
    names.extend('docs/exploration/'+name for name in (
        'topconf-downloads-20261117.json','latest-prior-art-20261117.json','merit-release-assets-20261117.json','source-publication-20261117.json',
        'selectivity-fusion-protocol-v1.json','selectivity-fusion-freeze-observation.json',
        'semantic-transport-protocol.json','semantic-transport-sigir.json',
        'semantic-transport-nips.json','crossencoder-results.json',
        'fixed-fusion-protocol.json','fixed-fusion-results.json',
        'semantic-transport-sigir-analysis.json','semantic-transport-nips-analysis.json',
        'teacher-certified-topk.json','demand-ranked-source-controls.json',
        'current-evidence.json'))
    files={name:(root/name).read_bytes() for name in sorted(set(names))}
    language_pattern=re.compile(r'[\u2e80-\u2fdf\u3000-\u303f\u3100-\u312f\u31a0-\u31bf\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef\U00020000-\U0003134f]')
    for name,body in files.items():
        if language_pattern.search(name) or language_pattern.search(body.decode('utf-8')):
            raise ValueError('Non-English text in source release: '+name)
    manifest=dict(repository_language='English; language-specific fixtures use Unicode escapes.',files={name:hashlib.sha256(body).hexdigest() for name,body in files.items()},
        scope='Source snapshot and the CC0 OpenAlex author-recovery proxy; not a complete enterprise experimental-data release.',
        exclusions=['credentials','raw enterprise/teacher data','model weights','other third-party corpora',
                    'model traces and quoted evidence','local virtual environments'],
        license_status='MIT for repository software; OpenAlex metadata and the author-recovery proxy are CC0; vendored unmodified ACM acmart retains its upstream LPPL license; other third-party datasets and literature retain their own terms and are excluded.')
    files['RELEASE-MANIFEST.json']=json.dumps(manifest,indent=2).encode()
    target=Path(output);target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,body in sorted(files.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o644<<16
            archive.writestr(info,body)
    return dict(path=str(target),files=len(files),sha256=hashlib.sha256(target.read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='release/expert-find-agent-source.zip')
    print(json.dumps(build(parser.parse_args().output),indent=2))
