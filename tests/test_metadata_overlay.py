import json
import tempfile
import unittest
from pathlib import Path
from expert_search.compiled_expert_search import corpus_snapshot
from expert_search.metadata.papers import publication_interval
from expert_search.metadata_overlay import MetadataOverlayPipeline


class MetadataOverlayTest(unittest.TestCase):
    def test_bind_text_owner_and_metadata_without_changing_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'base';enriched=Path(tmp)/'enriched';base.mkdir();enriched.mkdir()
            expert={'id':'a'};source={'id':'s','expert_id':'a','kind':'paper','text':'method'}
            extra={'publication':publication_interval('2021'),'contribution_status':'unverified','authorship_association':'provided_teacher_id_not_independently_verified'}
            for path in (base,enriched):(path/'experts.jsonl').write_text(json.dumps(expert)+'\n')
            (base/'evidence.jsonl').write_text(json.dumps(source)+'\n');(enriched/'evidence.jsonl').write_text(json.dumps(dict(source,**extra))+'\n')
            class Ranker: snapshot=corpus_snapshot(base)
            class Pipeline:
                ranker=Ranker()
                def search(self,*args,**kwargs):return {'source_snapshot_sha256':self.ranker.snapshot,'results':[{'expert':expert,'evidence':[source]}]}
                search_reranked=search
            pipeline=Pipeline();overlay=MetadataOverlayPipeline(pipeline,base,enriched);result=overlay.search('q')
            self.assertEqual(result['results'][0]['evidence'][0]['publication'],extra['publication'])
            self.assertNotIn('publication',source)
            source['text']='changed'
            with self.assertRaisesRegex(ValueError,'Changed source'):overlay.search('q')
            source['text']='method';(enriched/'evidence.jsonl').write_text(json.dumps(dict(source,text='other',**extra))+'\n')
            with self.assertRaisesRegex(ValueError,'Changed or duplicated'):MetadataOverlayPipeline(pipeline,base,enriched)

    def test_rejects_claimed_personal_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'base';enriched=Path(tmp)/'enriched';base.mkdir();enriched.mkdir()
            for path in (base,enriched):(path/'experts.jsonl').write_text('{"id":"a"}\n')
            row={'id':'s','expert_id':'a','text':'x'};(base/'evidence.jsonl').write_text(json.dumps(row)+'\n')
            (enriched/'evidence.jsonl').write_text(json.dumps(dict(row,contribution_status='verified'))+'\n')
            class Ranker: snapshot=corpus_snapshot(base)
            class Pipeline: ranker=Ranker()
            with self.assertRaisesRegex(ValueError,'Unverified role'):MetadataOverlayPipeline(Pipeline(),base,enriched)


if __name__=='__main__':unittest.main()
