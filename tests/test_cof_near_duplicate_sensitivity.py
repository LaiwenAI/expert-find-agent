import unittest
from experiments.cof_near_duplicate_sensitivity import near_title,title_tokens


class NearTitleTest(unittest.TestCase):
    def test_normalization_and_fixed_threshold(self):
        self.assertEqual(title_tokens('Fast, Expert SEARCH 2026!'),{'fast','expert','search','2026'})
        self.assertTrue(near_title('a b c d e','A B C D E f'))
        self.assertFalse(near_title('a b c d','a b c d'))
        self.assertFalse(near_title('a b c d e','a b c f g'))


if __name__=='__main__':unittest.main()
