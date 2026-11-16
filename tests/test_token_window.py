import unittest
from expert_search.token_window import TokenWindow


class TokenWindowTest(unittest.TestCase):
    def test_wait_tracks_usage_without_real_sleep(self):
        window=TokenWindow(100,60)
        window.record(10,60);window.record(20,25)
        self.assertEqual(window.delay(30,10),0)
        self.assertEqual(window.delay(30,40),40)
        self.assertEqual(window.delay(70,40),0)
        with self.assertRaises(ValueError):window.delay(70,101)
