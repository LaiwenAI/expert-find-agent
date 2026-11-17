import unittest
from expert_search.bilingual.numeric_word_audit import audit


class NumericWordAuditTest(unittest.TestCase):
    def test_flags_scaled_arabic_and_word_numbers(self):
        self.assertEqual(audit('5\u4e07\u5143','50,000 yuan')['status'],'uncertain')
        self.assertEqual(audit('\u4e09\u5e74','three years')['status'],'uncertain')
        self.assertEqual(audit('1000000','1 million')['status'],'uncertain')
        self.assertEqual(audit('\u767e\u5206\u4e4b\u4e94','five percent')['status'],'uncertain')

    def test_no_promotions_and_plain_numbers_remain_checked(self):
        self.assertEqual(audit('5 kg','6 kg')['status'],'mismatch')
        self.assertEqual(audit('5 kg','5 kg')['status'],'match')
        self.assertEqual(audit('10 kg','ten kilograms')['status'],'uncertain')
        self.assertEqual(audit('general research','general research')['status'],'match')


if __name__=='__main__':unittest.main()
