import hashlib
import unittest
from expert_search.partial_method_match import build_view,resolve_view


class PartialMethodTest(unittest.TestCase):
    def setUp(self):
        self.source=dict(id='s',expert_id='a',text='A documented research method.',kind='paper')
        self.candidates={'a':{'s':self.source},'b':{'z':dict(id='z',expert_id='b',text='Another source.',kind='paper')}}

    def test_foreign_card_and_cross_candidate_reference_rejected(self):
        card=dict(source_id='s',expert_id='wrong',source_sha256=hashlib.sha256(self.source['text'].encode()).hexdigest(),cards=[])
        with self.assertRaises(ValueError):build_view('Demand operation',self.candidates,{'s':card},'cards')
        _,mapping=build_view('Demand operation',self.candidates,{},'raw')
        rows=[dict(id='c0',l='t',d=['d0'],e=['c1:s0:p0'],m='A functional connection.',a=['Requires validation.'],g=['Other project requirements unknown.']),dict(id='c1',l='u',d=[],e=[],m='',a=[],g=['Unknown.'])]
        with self.assertRaises(ValueError):resolve_view(dict(experts=rows,read_sources=[]),mapping)

    def test_missing_cards_preserve_raw_context_and_unknown_is_not_a_negative(self):
        payload,mapping=build_view('Demand operation',self.candidates,{},'cards')
        self.assertTrue(all(s['raw_supplied'] for c in payload['candidates'] for s in c['sources']))
        rows=[dict(id=c,l='u',d=[],e=[],m='',a=[],g=['Missing support in supplied snapshot.']) for c in mapping['experts']]
        r=resolve_view(dict(experts=rows,read_sources=[]),mapping)
        self.assertEqual([a['level'] for a in r['ranking']],['insufficient','insufficient'])
        self.assertTrue(all(a['scope']=='partial_method_fit_only' for a in r['ranking']))

class PartialMethodFailureCategoryTest(unittest.TestCase):
    def test_categories_do_not_emit_model_data(self):
        import json
        from expert_search.partial_method_match import failure_category
        self.assertEqual(failure_category(ValueError('Missing full-demand gap')),'demand_gap')
        self.assertEqual(failure_category(ValueError('private arbitrary text')),'other_value_error')
        self.assertEqual(failure_category(json.JSONDecodeError('error','private',0)),'invalid_json')
        self.assertEqual(failure_category(TimeoutError()),'non_validation_error')

class PartialMethodMultilingualBudgetTest(unittest.TestCase):
    def test_long_complete_english_is_accepted_only_with_explicit_budget(self):
        from expert_search.partial_method_match import build_view,resolve_view
        source={'id':'s','expert_id':'a','text':'A complete source method.','kind':'paper'}
        _,mapping=build_view('Demand operation',{'a':{'s':source}},{},'raw')
        row={'id':'c0','l':'t','d':['d0'],'e':['c0:s0:p0'],'m':'method '+('x'*130),'a':['validate '+('x'*85)],'g':['gap '+('x'*105)]}
        response={'experts':[row],'read_sources':[]}
        with self.assertRaises(ValueError):resolve_view(response,mapping)
        accepted=resolve_view(response,mapping,{'mechanism':480,'assumption':240,'gap':300})
        self.assertEqual(accepted['ranking'][0]['level'],'transferable_method')
        row['m']='x'*481
        with self.assertRaises(ValueError):resolve_view(response,mapping,{'mechanism':480,'assumption':240,'gap':300})

class PartialMethodSchemaDiagnosticsTest(unittest.TestCase):
    def test_counts_field_defects_without_copying_text(self):
        from expert_search.partial_method_match import schema_diagnostics
        mapping={'experts':{'c0':'a'}};limits={'mechanism':4,'assumption':4,'gap':4}
        response={'experts':[{'id':'c0','l':'t','m':'secret','a':[],'g':['secret']} ]}
        result=schema_diagnostics(response,mapping,limits)
        self.assertEqual(result['mechanism_too_long'],1);self.assertEqual(result['gap_too_long'],1)
        self.assertEqual(result['missing_positive_assumption'],1);self.assertNotIn('secret',repr(result))
