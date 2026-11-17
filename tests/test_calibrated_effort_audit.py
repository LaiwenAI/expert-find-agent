import json
import tempfile
import unittest
from pathlib import Path
from experiments.bilingual.adaptive_effort_diagnostic import sha
from experiments.bilingual.audit_calibrated_effort import audit


class CalibrationAuditTest(unittest.TestCase):
    def test_same_algorithm_and_modified_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ids=[str(i) for i in range(10)];wrong=ids[:9]+['other'];src=['s'+x for x in ids]
            rows=[]
            for group in ('a','b'):
                for language in ('zh','en'):
                    arms={'exact':{'ids':ids,'sources':src,'seconds':.1},'high':{'ids':ids,'sources':src,'seconds':.01},'low':{'ids':wrong,'sources':src,'seconds':.001,'gap':.05}}
                    row={'id':group,'group_id':group,'language':language,'arms':arms}
                    if group=='b':row['actual_gate']={'seconds':.01,'escalated':False}
                    rows.append(row)
            protocol={'calibration_pairs':1,'evaluation_pairs':1,'retention_target':.995,'thresholds':[.1]}
            high={'views':2,'mean_retention':1.0,'ordered_equal':2,'simulated_mean_seconds':.01,'escalated':0}
            gated={'views':2,'mean_retention':1.0,'ordered_equal':2,'simulated_mean_seconds':.011,'escalated':2}
            calibration={'target_met':True,'threshold':None,'calibration':[{'threshold':None,**high},{'threshold':.1,**gated}]}
            for name,value in [('protocol.json',protocol),('calibration.json',calibration),('results.json',rows)]:
                (root/name).write_text(json.dumps(value))
            summary={'protocol_sha256':sha(root/'protocol.json'),'calibration_sha256':sha(root/'calibration.json'),'results_sha256':sha(root/'results.json'),
                     'selected_threshold':None,'evaluation_selected_simulated':high,'evaluation_high':high,'actual_results_match_frozen':2}
            (root/'summary.json').write_text(json.dumps(summary))
            self.assertTrue(audit(root)['same_algorithm_as_high_baseline'])
            calibration['threshold']=.1;(root/'calibration.json').write_text(json.dumps(calibration));summary['calibration_sha256']=sha(root/'calibration.json');(root/'summary.json').write_text(json.dumps(summary))
            with self.assertRaises(ValueError):audit(root)


if __name__=='__main__':unittest.main()
