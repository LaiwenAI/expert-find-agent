"""Package only frozen, hash-verified local evaluation members; never publish."""
import hashlib
import json
from pathlib import Path
import zipfile


def build():
    root=Path('artifacts/dataset-release/fresh-enterprise-eval-v1')
    manifest_path=root/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    verification=json.loads((root/'verification.json').read_text())
    digest=lambda data:hashlib.sha256(data).hexdigest()
    if verification['manifest_sha256']!=digest(manifest_path.read_bytes()):
        raise ValueError('Verification does not match frozen manifest')
    names=sorted([*manifest['files_sha256'],'manifest.json','verification.json'])
    output=Path('release/fresh-enterprise-eval-v1-local.zip')
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            if Path(name).name!=name:raise ValueError('Unsafe member name')
            body=(root/name).read_bytes()
            expected=manifest['files_sha256'].get(name)
            if expected and digest(body)!=expected:raise ValueError('Changed member: '+name)
            info=zipfile.ZipInfo(root.name+'/'+name,date_time=(2026,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o600<<16
            archive.writestr(info,body)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:raise ValueError('Archive CRC failure')
        for name in names:
            if digest(archive.read(root.name+'/'+name))!=digest((root/name).read_bytes()):
                raise ValueError('Archive member mismatch')
    record=dict(path=str(output),files=len(names),bytes=output.stat().st_size,
                sha256=digest(output.read_bytes()),dataset_manifest_sha256=verification['manifest_sha256'],
                status='Local only. Not public redistribution; no independent human gold labels.',
                verification_scope='Member hashes and ZIP round-trip, not semantic label correctness.')
    Path('artifacts/fresh-evaluation-package.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))


if __name__=='__main__':build()
