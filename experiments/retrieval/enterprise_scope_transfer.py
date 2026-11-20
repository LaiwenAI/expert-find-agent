"""Train a lightweight clause-role head with disjoint layouts and topics, with paired languages."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.enterprise_qualification_student import TOPICS, render, target_index
from experiments.retrieval.enterprise_requirement_roles import ScopedReader, prompt_data
from experiments.retrieval.enterprise_role_first_query import require_cuda
from experiments.retrieval.enterprise_routed_review import digest, write

ROLES = ("prior_qualification", "future_work", "operating_condition", "other")
FAMILIES = {'train_heading': (0,
                   [('Applicants must document completed work in the following activity.',
                     '\u7533\u8bf7\u4eba\u987b\u63d0\u4ea4\u4e0b\u5217\u5de5\u4f5c\u5df2\u5b8c\u6210\u7684\u8bc1\u660e\u3002'),
                    ('The awarded team will carry out the following activity.', '\u83b7\u9009\u56e2\u961f\u5c06\u5728\u9879\u76ee\u4e2d\u5b8c\u6210\u4e0b\u5217\u5de5\u4f5c\u3002'),
                    ('The existing site process used by all teams includes the following activity.',
                     '\u9879\u76ee\u5f00\u5c55\u65f6\u9700\u517c\u5bb9\u73b0\u573a\u5df2\u8fd0\u884c\u7684\u4e0b\u5217\u5de5\u5e8f\u3002'),
                    ('A former supplier reported the following activity in its own case study.',
                     '\u4e0b\u5217\u5de5\u4f5c\u4ec5\u89c1\u4e8e\u4e0a\u4e00\u5bb6\u4f9b\u5e94\u5546\u7684\u6848\u4f8b\u8bf4\u660e\u3002')]),
 'train_disposition': (0,
                       [('Evidence that the bidder already performed the following is required at '
                         'entry.',
                         '\u6295\u6807\u65f6\u5fc5\u987b\u8bc1\u660e\u6295\u6807\u65b9\u5df2\u5b8c\u6210\u4e0b\u5217\u5de5\u4f5c\u3002'),
                        ('The bidder must perform the following after the contract starts.',
                         '\u5408\u540c\u751f\u6548\u540e\uff0c\u6295\u6807\u65b9\u987b\u6267\u884c\u4e0b\u5217\u5de5\u4f5c\u3002'),
                        ('The project must accommodate this part of the plant baseline.',
                         '\u9879\u76ee\u5fc5\u987b\u517c\u5bb9\u5de5\u5382\u57fa\u7ebf\u4e2d\u7684\u4ee5\u4e0b\u73af\u8282\u3002'),
                        ('Below is an example of work performed by a different bidder.',
                         '\u4ee5\u4e0b\u5185\u5bb9\u662f\u5176\u4ed6\u6295\u6807\u65b9\u65e2\u5f80\u5de5\u4f5c\u7684\u793a\u4f8b\u3002')]),
 'calibration_attachment': (1,
                            [('Submit a record attributable to the proposing team for the activity '
                              'below.',
                              '\u4e0b\u5217\u4e8b\u9879\u987b\u63d0\u4f9b\u53ef\u5f52\u5c5e\u4e8e\u7533\u62a5\u56e2\u961f\u7684\u65e2\u5f80\u5de5\u4f5c\u8bb0\u5f55\u3002'),
                             ('Budget for the selected team to execute the activity below during '
                              'delivery.',
                              '\u9884\u7b97\u4e2d\u5e94\u5b89\u6392\u4e2d\u9009\u56e2\u961f\u5728\u4ea4\u4ed8\u671f\u95f4\u6267\u884c\u4e0b\u5217\u4e8b\u9879\u3002'),
                             ('Account for the activity below, which is already part of the host site.',
                              '\u4e0b\u5217\u4e8b\u9879\u5df2\u662f\u4f7f\u7528\u5355\u4f4d\u73b0\u573a\u7684\u4e00\u90e8\u5206\uff0c\u65b9\u6848\u9700\u4e88\u4ee5\u517c\u5bb9\u3002'),
                             ('For reference, a partner completed the activity below independently.',
                              '\u5408\u4f5c\u4f19\u4f34\u66fe\u72ec\u7acb\u5b8c\u6210\u4e0b\u5217\u4e8b\u9879\uff0c\u672c\u6bb5\u4ec5\u4f9b\u53c2\u8003\u3002')]),
 'held_earlier_conflict': (0,
                           [('For this item, evaluation requires evidence of previous work by the '
                             'applicant.',
                             '\u672c\u9879\u8bc4\u5ba1\u4ee5\u7533\u8bf7\u4eba\u65e2\u5f80\u4f5c\u54c1\u7684\u53ef\u6838\u9a8c\u8bc1\u660e\u4e3a\u4f9d\u636e\u3002'),
                            ('For this item, evaluation checks what the selected team delivers at '
                             'project close.',
                             '\u672c\u9879\u8003\u6838\u7684\u662f\u4e2d\u9009\u56e2\u961f\u7ed3\u9879\u65f6\u63d0\u4ea4\u7684\u65b0\u6210\u679c\u3002'),
                            ('For this item, the installed line and its existing checks are the '
                             'starting conditions.',
                             '\u672c\u9879\u4ee5\u5df2\u5b89\u88c5\u4ea7\u7ebf\u53ca\u5df2\u6709\u68c0\u6d4b\u73af\u8282\u4e3a\u73b0\u6709\u6761\u4ef6\u3002'),
                            ('For this item, the quoted work belongs to the issuer and is included for '
                             'background.',
                             '\u672c\u9879\u6240\u5f15\u5de5\u4f5c\u7531\u53d1\u5e03\u5355\u4f4d\u5b8c\u6210\uff0c\u5217\u5165\u6587\u4e2d\u4ec5\u7528\u4e8e\u8bf4\u660e\u80cc\u666f\u3002')]),
 'held_later_binding': (2,
                        [('The preceding item must be shown in the applicant\u2019s completed-work record.',
                          '\u4e0a\u4e00\u9879\u987b\u5728\u7533\u8bf7\u4eba\u7684\u5df2\u5b8c\u5de5\u8bb0\u5f55\u4e2d\u63d0\u4f9b\u8bc1\u660e\u3002'),
                         ('The preceding item is part of the new scope accepted at handover.',
                          '\u4e0a\u4e00\u9879\u5c5e\u4e8e\u672c\u6b21\u65b0\u589e\u4efb\u52a1\uff0c\u5728\u4ea4\u4ed8\u65f6\u9a8c\u6536\u3002'),
                         ('The preceding item describes the host unit\u2019s existing operating setup.',
                          '\u4e0a\u4e00\u9879\u8bf4\u660e\u4f7f\u7528\u5355\u4f4d\u5f53\u524d\u8fd0\u884c\u7684\u5de5\u51b5\u3002'),
                         ('The preceding item is an illustration from another team\u2019s dated report.',
                          '\u4e0a\u4e00\u9879\u662f\u53e6\u4e00\u56e2\u961f\u7559\u5b58\u62a5\u544a\u4e2d\u7684\u793a\u4f8b\u3002')])}
NEUTRAL = (("Personnel, paperwork and contacts follow.", "Details for the current item follow."),
           ("\u4eba\u5458\u3001\u6750\u6599\u548c\u8054\u7cfb\u65b9\u5f0f\u53e6\u5217\u3002", "\u4ee5\u4e0b\u8bf4\u660e\u672c\u9879\u8be6\u7ec6\u5185\u5bb9\u3002"))
DISTRACTOR = ("Earlier example: the applicant documented a different completed activity.",
              "\u524d\u9879\u4e3e\u4f8b\uff1a\u7533\u8bf7\u4eba\u5df2\u4e3a\u53e6\u4e00\u9879\u5b8c\u5de5\u6d3b\u52a8\u63d0\u4ea4\u4e86\u8bb0\u5f55\u3002")
MODES = ("target", "local", "full", "full_pair")


def make_dataset():
    panels = dict(train=[], calibration=[], held=[])
    for index, topic in enumerate(TOPICS):
        split = "train" if index < 6 else "calibration" if index < 9 else "held"
        for family, (mode, conditions) in FAMILIES.items():
            if not family.startswith(split + "_"):
                continue
            for language, target in enumerate(topic):
                group = f"{family}-{index}-{language}"
                filler, nearby = NEUTRAL[language]
                for role, binding in zip(ROLES, conditions, strict=True):
                    governing = binding[language]
                    if family == "held_earlier_conflict":
                        lines = [DISTRACTOR[language], filler, governing, nearby, filler, target]
                    elif mode == 2:
                        lines = [DISTRACTOR[language], filler, target, nearby, filler, governing]
                    elif mode == 1:
                        lines = [governing, nearby, filler, target]
                    else:
                        lines = [governing, target, filler]
                    item = dict(group=group, family=family, topic=index,
                                language=("en", "zh")[language], target=target, role=role, text="\n".join(lines))
                    target_index(item)
                    panels[split].append(item)
    return panels


def vector_inputs(rows, mode):
    outputs = []
    for row in rows:
        clauses, index = target_index(row)
        if mode == "full":
            payload = prompt_data(row["text"], clauses, index, full=True)
            outputs.append(json.dumps(payload, ensure_ascii=False))
        else:
            outputs.extend(render(row["text"], clauses, [index], "target_only" if mode == "target" else "scope"))
    return outputs


def classifier(x, y, groups, paired):
    x = torch.as_tensor(x, dtype=torch.float32, device="cuda")
    labels = torch.as_tensor(y, dtype=torch.float32, device="cuda") * 2 - 1
    center = x.mean(dim=0)
    intercept = labels.mean()
    design = x - center
    response = labels - intercept
    if paired:
        contrasts = [(i, j) for i, group in enumerate(groups) if y[i]
                     for j, other in enumerate(groups) if group == other and not y[j]]
        if not contrasts:
            raise ValueError("Pair training needs positive and negative views within a group")
        left, right = zip(*contrasts, strict=True)
        difference = x[list(left)] - x[list(right)]
        design = torch.cat((design, difference))
        response = torch.cat((response, labels[list(left)] - labels[list(right)]))
    alpha = torch.linalg.solve(design @ design.T + .10 * torch.eye(len(design), device="cuda"), response)
    return (design.T @ alpha).cpu().numpy(), center.cpu().numpy(), float(intercept)


def summary(rows, predicted):
    stats = Counter()
    by_role = defaultdict(Counter)
    groups = defaultdict(list)
    for row, prediction in zip(rows, predicted, strict=True):
        actual = row["role"] == "prior_qualification"
        stats.update(cases=1, true_positive=bool(actual and prediction), false_positive=bool(not actual and prediction),
                     false_negative=bool(actual and not prediction), correct=bool(actual == prediction))
        by_role[row["role"]].update(cases=1, flagged=bool(prediction))
        groups[row["group"]].append(actual == prediction)
    return dict(stats, all_correct_groups=sum(all(group) for group in groups.values()),
                groups=len(groups), by_role=dict(by_role))


def read_4b(reader, rows, full):
    predicted = []
    times = []
    for row in rows:
        clauses, index = target_index(row)
        start = perf_counter()
        result, _ = reader.judge(row["text"], clauses, [index], full)
        times.append((perf_counter() - start) * 1000)
        predicted.append(result[index][0] == 2)
    return dict(summary(rows, predicted), p95_ms=float(np.quantile(times, .95)))


def run(args):
    require_cuda(args.output, args.dataset)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = make_dataset()
    labels = {key: np.asarray([r["role"] == ROLES[0] for r in rows]) for key, rows in data.items()}
    groups = [r["group"] for r in data["train"]]
    embedding_revision = verify_checkpoint(args.embedding)["revision"]
    encoder = QwenTorchEmbedding(args.embedding)
    features = {mode: {key: encoder.encode(vector_inputs(rows, mode)) for key, rows in data.items()}
                for mode in ("target", "local", "full")}
    report = {}
    for mode in MODES:
        inputs = mode.removesuffix("_pair")
        started = perf_counter()
        weight, center, intercept = classifier(features[inputs]["train"], labels["train"], groups, mode == "full_pair")
        fit_ms = (perf_counter() - started) * 1000
        calibration = (features[inputs]["calibration"] - center) @ weight + intercept
        errors = np.abs(calibration[(calibration > 0) != labels["calibration"]])
        cutoff = float(errors.max()) if len(errors) else 0.
        held = (features[inputs]["held"] - center) @ weight + intercept
        high_confidence = np.abs(held) > cutoff
        report[mode] = dict(
            **summary(data["held"], held > 0), confidence_cutoff=cutoff,
            confident=int(high_confidence.sum()),
            confident_errors=int((high_confidence & ((held > 0) != labels["held"])).sum()),
            fit_ms=fit_ms,
        )
    reader = ScopedReader(args.reader)
    report["4b_local"] = read_4b(reader, data["held"], False)
    report["4b_full"] = read_4b(reader, data["held"], True)
    write(args.dataset, data)
    write(args.output, dict(
        panels={key: len(rows) for key, rows in data.items()}, methods=report,
        data_sha256=digest(args.dataset), script_sha256=digest(Path(__file__)),
        embedding_revision=embedding_revision,
        reader_revision=reader.provenance["revision"], gpu=torch.cuda.get_device_name(),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("embedding", "reader", "output", "dataset"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
