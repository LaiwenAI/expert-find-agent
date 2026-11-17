import unittest
from expert_search.offpolicy import evaluate


class OffPolicyTest(unittest.TestCase):
    def test_ips_and_effective_sample_size(self):
        events=[{'event_id':'1','context':'x','arm':'a','outcome':1.,'logging_probabilities':{'a':.5,'b':.5}},
                {'event_id':'2','context':'x','arm':'b','outcome':0.,'logging_probabilities':{'a':.5,'b':.5}}]
        result=evaluate(events,{'x':{'a':1.,'b':0.}})
        self.assertEqual(result['ips'],1.)
        self.assertEqual(result['snips'],1.)
        self.assertEqual(result['effective_sample_size'],1.)

    def test_missing_rewards_and_unsupported_target_are_rejected(self):
        row={'event_id':'1','context':'x','arm':'a','outcome':None,'logging_probabilities':{'a':1.,'b':0.}}
        with self.assertRaisesRegex(ValueError,'Missing outcome'):evaluate([row],{'x':{'a':1.,'b':0.}})
        row['outcome']=1.
        with self.assertRaisesRegex(ValueError,'lacks logging support'):evaluate([row],{'x':{'a':0.,'b':1.}})
        with self.assertRaises(ValueError):evaluate([row,row],{'x':{'a':1.,'b':0.}})

    def test_zero_realized_weights_are_reported(self):
        row={'event_id':'1','context':'x','arm':'a','outcome':1.,'logging_probabilities':{'a':.5,'b':.5}}
        result=evaluate([row],{'x':{'a':0.,'b':1.}})
        self.assertIsNone(result['snips'])
        self.assertEqual(result['effective_sample_size'],0.)

if __name__=='__main__':unittest.main()
