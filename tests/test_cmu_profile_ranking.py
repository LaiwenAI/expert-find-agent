import numpy as np

from experiments.decision.cmu_profile_ranking import grouped_folds, make_features, preference_data


def test_folds_keep_title_aliases_and_reviewers_together():
    cases = [
        ("first", "a", ()), ("second", "b", ()), ("first", "c", ()), ("second", "d", ()),
        *((f"other-{i}", f"p-{i}", ()) for i in range(6)),
    ]
    queries = {"a": "Repeated title\nOne abstract", "b": "Repeated Title!\nAnother abstract",
               "c": "First only", "d": "Second only"} | {f"p-{i}": f"Topic {i}" for i in range(6)}
    folds = grouped_folds(cases, queries)
    component = next(group for fold in folds for group in fold if 0 in group)
    assert component == [0, 1, 2, 3]
    assert sorted(i for fold in folds for group in fold for i in group) == list(range(len(cases)))


def test_preference_examples_omit_ties_and_mirror_order():
    cases = [("r", "a", ()), ("r", "b", ()), ("r", "c", ())]
    labels = {"r": {"a": 1, "b": 5, "c": 1}}
    features = {i: np.array([value]) for i, value in enumerate((1, 5, 3))}
    pair_data, classes = preference_data(range(3), features, cases, labels)
    assert pair_data[:, 0].tolist() == [-4, 4, 2, -2]
    assert classes.tolist() == [0, 1, 1, 0]


def test_same_dense_profile_gets_more_relative_credit_against_weaker_background():
    query = np.array([1., 0.])
    owned = np.array([[.9, 0.], [.8, 0.]])
    center = query.copy()
    strong = make_features(owned, center, query, np.array([[.9, 0.], [.95, 0.]]))
    weak = make_features(owned, center, query, np.array([[.1, 0.], [.15, 0.]]))
    assert strong[:4] == weak[:4]
    assert strong[5] < weak[5]


def test_merit_transfer_excludes_source_overlap_and_conflicting_views():
    from types import SimpleNamespace

    from experiments.decision.cmu_merit_transfer import metrics, prepare_external, title

    def row(query, docs, label):
        return SimpleNamespace(query=query, papers=docs, label=label)

    rows = [
        row("Shared title", ("Source One", "Source Two"), 1),
        row("Target", ("Source One", "Source Two"), 1),
        row("Target", ("General One", "General Two"), 0),
        row("Target", ("General Two", "General One"), 1),
        row("Target", ("Other One", "Other Two"), 0),
    ]
    retained, excluded, conflicts = prepare_external(rows, {title("Shared title")}, set())
    assert (len(retained), excluded, conflicts) == (2, 1, 1)
    result = metrics(retained, [float(item[3]) for item in retained])
    assert result["within_target_pairs"] == result["mixed_targets"] == 1
    assert result["within_target_loss"] == 0
