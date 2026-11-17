import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from expert_search.citation_overlay import CitationOverlayPipeline
from expert_search.compiled_expert_search import corpus_snapshot
from expert_search.metadata.papers import citation_metadata


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class CitationOverlayTest(unittest.TestCase):
    def test_traceability_and_unknowns(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'base';sidecar=Path(tmp)/'sidecar';base.mkdir();sidecar.mkdir()
            raw={'teacher_id':'a','title':'Title','gs_abstract':'Method','gs_scholarly_total_citations':7}
            source=Path(tmp)/'papers.jsonl';source.write_text(json.dumps(raw)+'\n')
            paper={'id':'paper:1','expert_id':'a','kind':'paper','title':'Title','text':'Title\nMethod'}
            (base/'experts.jsonl').write_text('{"id":"a"}\n');(base/'evidence.jsonl').write_text(json.dumps(paper)+'\n')
            row={'source_id':'paper:1','expert_id':'a','raw_line':1,'raw_record_sha256':hashlib.sha256(json.dumps(raw,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),'citations':citation_metadata(raw)}
            metadata=sidecar/'papers.jsonl';metadata.write_text(json.dumps(row)+'\n')
            (sidecar/'manifest.json').write_text(json.dumps({'source_sha256':sha(source),'metadata_sha256':sha(metadata),'rows':1}))
            class Ranker: snapshot=corpus_snapshot(base)
            class Pipeline:
                ranker=Ranker()
                def search(self,*args,**kwargs):return {'source_snapshot_sha256':self.ranker.snapshot,'results':[{'expert':{'id':'a'},'evidence':[paper]}]}
            overlay=CitationOverlayPipeline(Pipeline(),base,sidecar,source);item=overlay.search('x')['results'][0]['evidence'][0]
            self.assertEqual(item['citation_context']['observed_total'],7)
            self.assertIsNone(item['citation_context']['observation_date']);self.assertIsNone(item['citation_context']['personal_credit'])
            self.assertNotIn('citation_context',paper)
            source.write_text(json.dumps(dict(raw,teacher_id='b'))+'\n')
            with self.assertRaisesRegex(ValueError,'Changed citation'):CitationOverlayPipeline(Pipeline(),base,sidecar,source)


if __name__=='__main__':unittest.main()
