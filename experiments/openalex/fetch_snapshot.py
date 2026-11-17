"""Fetch a small official OpenAlex API response snapshot for offline construction."""
import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime,timezone
from pathlib import Path


SEARCHES=('robotic control','battery materials','wastewater treatment','advanced manufacturing','solar energy')
PERIODS={'history':('2010-01-01','2019-12-31'),'target':('2020-01-01','2023-12-31')}
FIELDS='id,title,publication_year,authorships,language'


def main():
    root=Path('artifacts/public/openalex-api-snapshot-v1');root.mkdir(parents=True,exist_ok=False)
    records=[]
    for i,query in enumerate(SEARCHES):
        for period,(start,end) in PERIODS.items():
            params={'search':query,'filter':f'from_publication_date:{start},to_publication_date:{end}',
                    'per-page':200,'select':FIELDS}
            url='https://api.openalex.org/works?'+urllib.parse.urlencode(params)
            request=urllib.request.Request(url,headers={'User-Agent':'expert-search-research/0.1'})
            response=urllib.request.urlopen(request,timeout=45)
            if response.headers.get_content_type()!='application/json':raise ValueError('Non-JSON API response')
            raw=response.read();parsed=json.loads(raw)
            if not isinstance(parsed.get('results'),list):raise ValueError('Invalid API response')
            name=f'{i:02d}-{period}.json';(root/name).write_bytes(raw)
            records.append({'file':name,'url':url,'sha256':hashlib.sha256(raw).hexdigest(),'count':len(parsed['results'])})
            print(name,len(parsed['results']),flush=True);time.sleep(.25)
    manifest={'fetched_utc':datetime.now(timezone.utc).isoformat(),'records':records,
              'scope':'Dynamic API result snapshot, not a fixed OpenAlex monthly release. Raw responses may contain extra author metadata and are kept locally; only selected IDs and titles enter the derived benchmark.'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
