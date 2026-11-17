import unittest
from expert_search.metadata.papers import publication_interval,availability,paper_metadata


class PaperMetadataTest(unittest.TestCase):
    def test_partial_dates_not_invented_day_and_future_cutoff(self):
        year=publication_interval('2026');self.assertEqual(year['precision'],'year')
        self.assertEqual(availability(year,'2026-06-01'),'overlaps_cutoff_date_precision_uncertain')
        self.assertEqual(availability(publication_interval('2026/5'),'2025-12-31'),'after_cutoff')
        self.assertEqual(availability(publication_interval('2024/2/29'),'2024-02-29'),'before_or_on_cutoff')
        self.assertEqual(publication_interval('2025/2/29')['status'],'invalid_calendar_date')

    def test_high_citations_do_not_create_individual_credit(self):
        r=paper_metadata(dict(gs_publication_date='2020',gs_authors='Author A, Author B',gs_scholarly_total_citations=10000,gs_scholarly_citations_by_year=[dict(year=2024,citations=10001)]))
        self.assertIsNone(r['contribution']['personal_method_credit'])
        self.assertEqual(r['contribution']['verified_roles'],[])
        self.assertEqual(r['citations']['total_history_status'],'yearly_sum_exceeds_total')
        self.assertIsNone(r['citations']['observation_date'])
