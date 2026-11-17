"""Controlled request-grounding contrasts; labels are never model inputs."""


def cases():
    # Distinct English templates isolate object, threshold and added constraint.
    templates = [
        ("Find a method to measure water flow.", "Find a method to measure oil flow."),
        ("Find a sensor operating at temperatures above 80 C.", "Find a sensor operating at temperatures below 80 C."),
        ("Find a method for detecting defects in steel sheets.", "Find a method for detecting defects in steel sheets at 100 sheets per minute."),
        ("Find a process to remove salt from water.", "Find a process to remove salt from water without electricity."),
        ("Find a method to classify plant diseases from images.", "Find a method to classify animal diseases from images."),
        ("Find a battery material with capacity at least 200 mAh/g.", "Find a battery material with capacity at least 300 mAh/g."),
    ]
    result = []
    for index, (source, altered) in enumerate(templates):
        for variant, derived, expected in (("faithful", source, "supported"), ("altered", altered, "changed_or_unsupported")):
            result.append({"id": f"heldout-{index}-{variant}", "source": source,
                           "derived": derived, "expected": expected, "template": index})
    return result


GROUNDING_INSTRUCTION = (
    "Judge whether the derived request is supported by the source request. "
    "A source request describes a need; it need not prove that a solution exists. "
    "A changed object, reversed condition, or added specific requirement is not supported. "
    "Use uncertain when the relationship cannot be determined from the supplied text."
)


def chinese_cases():
    templates = [('\u5bfb\u627e\u6d4b\u91cf\u6c34\u6d41\u7684\u65b9\u6cd5\u3002', '\u5bfb\u627e\u6d4b\u91cf\u6cb9\u6d41\u7684\u65b9\u6cd5\u3002'), ('\u5bfb\u627e\u5de5\u4f5c\u6e29\u5ea6\u9ad8\u4e8e80\u6444\u6c0f\u5ea6\u7684\u4f20\u611f\u5668\u3002', '\u5bfb\u627e\u5de5\u4f5c\u6e29\u5ea6\u4f4e\u4e8e80\u6444\u6c0f\u5ea6\u7684\u4f20\u611f\u5668\u3002'), ('\u5bfb\u627e\u68c0\u6d4b\u94a2\u677f\u7f3a\u9677\u7684\u65b9\u6cd5\u3002', '\u5bfb\u627e\u6bcf\u5206\u949f\u68c0\u6d4b100\u5f20\u94a2\u677f\u7f3a\u9677\u7684\u65b9\u6cd5\u3002'), ('\u5bfb\u627e\u53bb\u9664\u6c34\u4e2d\u76d0\u5206\u7684\u65b9\u6cd5\u3002', '\u5bfb\u627e\u65e0\u9700\u7535\u529b\u53bb\u9664\u6c34\u4e2d\u76d0\u5206\u7684\u65b9\u6cd5\u3002'), ('\u5bfb\u627e\u57fa\u4e8e\u56fe\u50cf\u8bc6\u522b\u690d\u7269\u75be\u75c5\u7684\u65b9\u6cd5\u3002', '\u5bfb\u627e\u57fa\u4e8e\u56fe\u50cf\u8bc6\u522b\u52a8\u7269\u75be\u75c5\u7684\u65b9\u6cd5\u3002'), ('\u5bfb\u627e\u5bb9\u91cf\u81f3\u5c11200 mAh/g\u7684\u7535\u6c60\u6750\u6599\u3002', '\u5bfb\u627e\u5bb9\u91cf\u81f3\u5c11300 mAh/g\u7684\u7535\u6c60\u6750\u6599\u3002')]
    result = []
    for index, (source, altered) in enumerate(templates):
        for variant, derived, expected in (("faithful", source, "supported"), ("altered", altered, "changed_or_unsupported")):
            result.append({"id": f"chinese-{index}-{variant}", "source": source, "derived": derived, "expected": expected})
    return result
