import unittest
from expert_search.transfer_case import resolve_transfer


class TransferCaseTest(unittest.TestCase):
    def setUp(self):
        self.job=dict(expert_id='e',demand='technical operation',sources={'s':dict(expert_id='e',text='actual method evidence')})
        self.ds={'d':dict(start=0,end=19,text='technical operation')}
        self.ss={'s':{'p':dict(start=0,end=22,text='actual method evidence')}}
        self.proposal=dict(feasible=True,capability='actual method evidence',target_operation='technical operation',mechanism='A conditional functional connection to the target operation.',demand_span_id='d',evidence_spans=[dict(evidence_id='s',evidence_span_id='p')],assumptions=['Target data availability remains unvalidated.'],remaining_demand_gaps=['Deployment and target performance are unknown.'])

    def test_retains_future_uncertainty_and_original_provenance(self):
        r=resolve_transfer(self.job,self.proposal,self.ds,self.ss)
        self.assertEqual(r['status'],'proposal_pending_audit')
        self.assertEqual(r['citations'][0]['quote'],'actual method evidence')
        self.assertEqual(r['assumptions'],self.proposal['assumptions'])

    def test_foreign_sources_and_unconditional_claims_rejected(self):
        for p in [dict(self.proposal,assumptions=[]),dict(self.proposal,remaining_demand_gaps=[]),dict(self.proposal,evidence_spans=[dict(evidence_id='foreign',evidence_span_id='p')])]:
            with self.assertRaises(ValueError):resolve_transfer(self.job,p,self.ds,self.ss)
