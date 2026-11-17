import tempfile
from pathlib import Path
import unittest
from expert_search.repository_text import escape_source_text,write_english_report


class RepositoryReportTextTest(unittest.TestCase):
    def test_source_characters_escape_without_losing_markdown_structure(self):
        source='## Evidence\n\u7814\u7a76: 5 mm\n'
        self.assertEqual(escape_source_text(source),'## Evidence\n\\u7814\\u7a76: 5 mm\n')
        self.assertEqual(escape_source_text('ASCII path/data.jsonl\n'),'ASCII path/data.jsonl\n')
        self.assertEqual(escape_source_text('B\u00fcttcher; \u03bc m'),'B\u00fcttcher; \u03bc m')

    def test_report_writer_creates_only_requested_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'reports/check.md'
            write_english_report(output,'Source: \u8bc1\u636e\n')
            self.assertTrue(output.exists())
            self.assertTrue(output.read_text().isascii())
            self.assertEqual(output.read_text().count('\n'),1)
