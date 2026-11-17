import unittest
from expert_search.bilingual.translation_anchors import protect, restore
from expert_search.bilingual.postvalidation import check_pair


class TranslationAnchorsTest(unittest.TestCase):
    def test_comparators_quantities_and_units_survive_restoration(self):
        source='厚度不低于１２．５毫米；浓度5 mM以下。'
        value=protect(source)
        self.assertEqual([r['english'] for r in value['spans']],['>= 12.5 mm','<= 5 mM'])
        result=restore('Thickness [[NUM000000]]; concentration [[NUM000001]].',value['spans'])
        self.assertEqual(check_pair(source,result)['status'],'match')
        self.assertEqual([source[r['start']:r['end']] for r in value['spans']],[r['source'] for r in value['spans']])

    def test_missing_duplicate_foreign_and_colliding_tokens_rejected(self):
        spans=protect('至少5 mm')['spans']
        for text in ['', '[[NUM000000]][[NUM000000]]','[[NUM000001]]','[[NUM000000]] [[NUMbad]]']:
            with self.assertRaises(ValueError):restore(text,spans)
        with self.assertRaises(ValueError):protect('Text [[NUM000000]]')

    def test_unrecognized_negation_is_not_silently_canonicalized(self):
        result=protect('not necessarily below 5 mm')
        self.assertEqual(result['spans'][0]['english'],'5 mm')
        self.assertIn('not necessarily below',result['masked'])
        self.assertEqual(check_pair('not necessarily below 5 mm',restore(result['masked'],result['spans']))['status'],'uncertain')

    def test_chemical_identifier_not_changed_and_unit_case_kept(self):
        result=protect('CO2浓度5 mM，尺寸5 mm')
        self.assertIn('CO2',result['masked'])
        self.assertEqual([r['english'] for r in result['spans']],['5 mM','5 mm'])

class GapAssemblyTest(unittest.TestCase):
    def test_constraints_do_not_depend_on_model_token_copying(self):
        from expert_search.bilingual.translation_anchors import split_gaps, assemble_gaps
        row=split_gaps('厚度不低于5毫米，浓度2 mM以下')
        self.assertEqual(row['gaps'],['厚度','，浓度',''])
        restored=assemble_gaps(['Thickness', '; concentration', ''],row['spans'])
        self.assertEqual(check_pair('厚度不低于5毫米，浓度2 mM以下',restored)['status'],'match')
        with self.assertRaises(ValueError):assemble_gaps(['Thickness 10','; concentration',''],row['spans'])
        with self.assertRaises(ValueError):assemble_gaps(['Thickness'],row['spans'])
