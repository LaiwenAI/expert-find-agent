import numpy as np
import pytest

pytest.importorskip("torch")

from experiments.credit.linked_witness_dispatch import output, request


def test_article_union_can_misattribute_roles_to_a_linked_contributor():
    rows = [(0,"linked",{"method"}),(0,"neighbor",{"analysis"}),
            (1,"linked",{"writing"}),(1,"other",{"software"})]
    requested = [(0,"analysis"),(1,"software")]
    assert output(rows,range(4),requested) == set()
    assert output(rows,range(4),requested,ignore_owner=True) == {"linked"}


def test_adaptive_dispatch_avoids_completing_a_falsified_owner():
    rows = [(0,"first",{"unrelated"}),(1,"first",{"writing"}),
            (0,"second",{"method","analysis"}),(1,"second",{"writing"})]
    requested = [(0,"analysis"),(0,"method"),(1,"writing")]
    scores = np.array([[.9,.9,0.],[0.,0.,.85],
                       [.8,.8,0.],[0.,0.,.8]])
    result, opened = request(rows,scores,requested,3,"adaptive")
    assert opened == [0,2,3]
    assert result == {"second"}
    assert request(rows,scores,requested,3,"owner_bottleneck")[0] == set()


def test_live_verification_uses_reader_decisions_instead_of_gold_roles():
    from experiments.credit.live_linked_witness_verification import verify

    class Reader:
        def score_pairs(self, pairs, *, task):
            return [-1. if doc == "reject" else 2. for _,doc in pairs], 0

    rows = [(0,"owner",{"analysis"}),(1,"owner",{"writing"})]
    requested = [(0,"analysis"),(1,"writing")]
    score = np.array([[.8,0.],[0.,.7]])
    found, opened, checked = verify(rows,score,["reject","accept"],["A","B"],requested,
                                    Reader(),"owner_mean",2,0.)
    assert not found
    assert len(opened) == 2
    assert len(checked) == 2 and sum(accepted for _,_,accepted in checked) == 1
