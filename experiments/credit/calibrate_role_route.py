"""Fit selective decoding thresholds using article-disjoint validation cases."""


MODES = ("margin", "agreement", "empty_only")


def should_accept(features, policy):
    threshold = policy["min_margin"]
    return (
        threshold is not None
        and features["distinct"]
        and features["margin"] >= threshold
        and (policy["mode"] != "agreement" or features["agreement"])
        and (policy["mode"] != "empty_only" or features["empty"])
    )


def calibrate(rows, mode):
    if mode not in MODES:
        raise ValueError(f"Unknown route calibration: {mode}")
    eligible = []
    for row in rows:
        fast, reader = (row["arms"][arm] for arm in ("ridge_embed", "ridge_reader"))
        features = fast.get("route_features")
        if features is None or not should_accept(features, dict(mode=mode, min_margin=0)):
            continue
        wanted = {(row["doi"], author) for author in row["wanted"]}
        added = set(map(tuple, fast["citations"])) - set(map(tuple, reader["citations"])) - wanted
        loses_success = reader["success"] and not fast["success"]
        eligible.append((features["margin"], bool(added or loses_success)))
    # Include all ties. Pick the widest region with no new unsupported
    # citation and no lost exact answer on the validation sample.
    last_error = max((margin for margin, error in eligible if error), default=-1.0)
    accepted = [margin for margin, _ in eligible if margin > last_error]
    return dict(mode=mode, min_margin=min(accepted) if accepted else None,
                eligible=len(eligible), accepted=len(accepted), validation_requests=len(rows))
