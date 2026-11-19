import numpy as np

from experiments.credit.jats_linked_gate import predict


def test_consensus_never_promotes_one_interpretation_over_another():
    registry = {"left": ({"owner"}, {0:{"owner"},1:{"owner"}}),
                "right": ({"owner"}, {2:{"owner"}})}
    scores = np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,.95]])
    found, decided, plans = predict(registry,("left","right"),[0,0,1],scores,.1)
    assert plans == {(0,1,2),(0,1,3)}
    assert not found and not decided
    point, point_decided, _ = predict(registry,("left","right"),[0,0,1],scores,None)
    assert point == {"owner"} and point_decided


def test_scope_and_owner_are_required_for_each_role():
    registry = {"left": ({"owner","other"}, {0:{"owner"},1:{"other"},2:{"owner"}}),
                "right": ({"owner","other"}, {1:{"owner"},2:{"owner"}})}
    scores = np.eye(3)
    found, decided, _ = predict(registry,("left","right"),[0,0,1],scores,None)
    assert not found and decided


def test_no_distinct_role_assignment_is_unresolved():
    registry = {"left": ({"owner"}, {0:{"owner"}}),"right": ({"owner"}, {0:{"owner"}})}
    found, decided, plans = predict(registry,("left","right"),[0,0,1],
                                    np.array([[1.,0.],[1.,0.],[1.,0.]]),None)
    assert not found and not decided and not plans
