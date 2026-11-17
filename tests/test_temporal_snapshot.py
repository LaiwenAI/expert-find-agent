import json
import tempfile
import unittest
from pathlib import Path
from expert_search.metadata.papers import publication_interval
from expert_search.metadata.temporal_snapshot import build_temporal_snapshot


class TemporalSnapshotTest(unittest.TestCase):
    def test_inclusive_cutoff_excludes_unknown_and_partial_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source'; source.mkdir(); target=Path(tmp)/'snapshot'
            (source/'experts.jsonl').write_text(json.dumps({'id':'e','institution':'current institution'})+'\n')
            data=[('old','paper','2019'),('same','paper','2020-06-10'),('partial','paper','2020'),
                  ('future','paper','2021'),('missing','paper',None),('profile','profile',None),('forged','paper','2022')]
            rows=[dict(id=i,expert_id='e',kind=kind,text=i,publication=publication_interval(raw)) for i,kind,raw in data]
            rows[-1]['publication']['start']='2018-01-01'
            rows[-1]['publication']['end_exclusive']='2019-01-01'
            rows[0]['citation_count']=999
            rows[0]['metadata']={'current_metric':123}
            (source/'evidence.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            summary=build_temporal_snapshot(source,target,'2020-06-10')
            self.assertEqual([json.loads(x)['id'] for x in (target/'evidence.jsonl').read_text().splitlines()],['old','same'])
            self.assertEqual((summary['kept_papers'],summary['excluded']),(2,5))
            self.assertEqual(json.loads((target/'experts.jsonl').read_text()), {'id':'e'})
            first=json.loads((target/'evidence.jsonl').read_text().splitlines()[0])
            self.assertNotIn('citation_count',first)
            self.assertNotIn('metadata',first)
            self.assertEqual(summary['counts']['overlaps_cutoff_date_precision_uncertain'],1)
            self.assertEqual(summary['counts']['unknown_publication_date'],1)
            self.assertEqual(summary['counts']['inconsistent_publication_interval'],1)
            with self.assertRaises(FileExistsError):build_temporal_snapshot(source,target,'2020-06-10')


if __name__=='__main__':unittest.main()
