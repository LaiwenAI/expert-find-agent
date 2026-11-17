# Historical Industrial-Demand Annotation Protocol

This is a historical research entry point. The [current English SIGIR main paper](paper/sigir/main.tex) defines the latest task, evidence boundaries, and results. Insufficient material is not a negative label for actual expert ability.

Source: `data/\u771f\u5b9e\u4f01\u4e1a\u9700\u6c42/\u5168\u56fd\u5927\u5b66\u751f\u521b\u4e1a\u670d\u52a1\u7f51_\u4ea7\u4e1a\u547d\u9898.xlsx`. Original-language path components are Unicode-escaped. A fixed random seed of 42 selects 20 records containing both a proposition title and content. The export concatenates the title and complete content; it does not automatically remove company introductions. Technical-condition extraction is a separate step. The original queries therefore cannot be described as excluding background or brand terms.

`PYTHONPATH=src python3 -m expert_search demands` exports queries, rankings from three baselines, and a union candidate pool for annotation. Candidates are sorted by `expert_id`; annotators do not see algorithm names or ranks. Evidence currently comes from the first retrieval method's top three records. A cross-method evidence union is still needed to control presentation bias.

At least two domain-knowledgeable annotators independently grade each demand/expert pair:

- **0:** the supplied evidence does not support relevant capability.
- **1:** the field is related, but direct evidence for the core technical capability is missing.
- **2:** evidence supports some important technical requirements.
- **3:** evidence supports the core need and application setting, with identifiable supporting passages.

Record sources, reasons, and missing information that prevents judgment. Use `null` for unknowns; do not automatically convert missing material into inadequate capability. Conflicts require a third adjudicator. Report weighted agreement and the adjudication fraction. Hide model recommendation explanations from annotators to avoid influencing their assessments.

Use graded nDCG@10. Treat `grade>=2` as relevant for MRR@10 and Recall@10. Experts outside the pool remain unjudged: recall is relative to the judged relevant pool, not full-corpus recall. Report query-paired bootstrap intervals and preserve algorithm versions, corpus SHA256, seeds, configuration, per-query rankings, and failures.

Human gold labels are not yet available. Model self-assessment cannot be described as human validation or used to establish real-demand accuracy. Report the paper-author recovery proxy separately.

A future project's requested performance or delivery target does not automatically require the candidate to have achieved it historically. Grade the documented technical foundation and match for undertaking the research. Record unknown prior integration experience as a gap instead of directly assigning zero potential relevance. Distinguish explicit personal qualifications from future technical goals.
