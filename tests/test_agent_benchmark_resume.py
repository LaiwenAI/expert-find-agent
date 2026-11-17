import json
import tempfile
import unittest
from pathlib import Path

from expert_search.agent_benchmark import run


class ResumeGuardTests(unittest.TestCase):
    def test_existing_result_without_code_fingerprint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark, output = root / 'benchmark', root / 'output'
            benchmark.mkdir()
            output.mkdir()
            (benchmark / 'manifest.json').write_text(json.dumps({'excluded_evidence': []}))
            (output / 'old.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'no code fingerprint'):
                run(str(benchmark), str(output), limit=1)
