# Historical Industrial-Demand Annotation Protocol

This is a historical research entry point. The [current English SIGIR main paper](paper/sigir/main.tex) defines the latest task, evidence boundaries, and results. Insufficient material is not a negative label for actual expert ability.

Source: `data/\u771f\u5b9e\u4f01\u4e1a\u9700\u6c42/\u5168\u56fd\u5927\u5b66\u751f\u521b\u4e1a\u670d\u52a1\u7f51_\u4ea7\u4e1a\u547d\u9898.xlsx`. Original-language path components are Unicode-escaped. A fixed random seed of 42 selects 20 records containing both a proposition title and content. The export concatenates the title and complete content; it does not automatically remove company introductions. Technical-condition extraction is a separate step. The original queries therefore cannot be described as excluding background or brand terms.

`PYTHONPATH=src python3 -m expert_search demands` exports queries, rankings from three baselines, and a union candidate pool for annotation. Candidates are sorted by `expert_id`; annotators do not see algorithm names or ranks. Evidence is the deduplicated, source-ID-ordered union of up to three returned records per candidate from each method; method scores are removed, and mismatched ownership or conflicting source snapshots are rejected. This reduces first-method-only presentation bias but does not ensure all relevant sources enter the pool; source text and expert names can reveal identities, and each returned source remains unverified. Previously exported packets retain their original policy and must not be silently relabeled as union-evidence packets.

At least two domain-knowledgeable annotators independently grade each demand/expert pair:

- **0:** the supplied evidence does not support relevant capability.
- **1:** the field is related, but direct evidence for the core technical capability is missing.
- **2:** evidence supports some important technical requirements.
- **3:** evidence supports the core need and application setting, with identifiable supporting passages.

Record sources, reasons, and missing information that prevents judgment. Use `null` for unknowns; do not automatically convert missing material into inadequate capability. Conflicts require a third adjudicator. Report weighted agreement and the adjudication fraction. Hide model recommendation explanations from annotators to avoid influencing their assessments.

Use graded nDCG@10. Treat `grade>=2` as relevant for MRR@10 and Recall@10. Experts outside the pool remain unjudged: recall is relative to the judged relevant pool, not full-corpus recall. Report query-paired bootstrap intervals and preserve algorithm versions, corpus SHA256, seeds, configuration, per-query rankings, and failures.

Human gold labels are not yet available. Model self-assessment cannot be described as human validation or used to establish real-demand accuracy. Report the paper-author recovery proxy separately.

A future project's requested performance or delivery target does not automatically require the candidate to have achieved it historically. Grade the documented technical foundation and match for undertaking the research. Record unknown prior integration experience as a gap instead of directly assigning zero potential relevance. Distinguish explicit personal qualifications from future technical goals.

## Bilingual intent-pair review

The local high-overlap packet contains two generated requests from the same original need, in both languages. It hides the intended type, previous screening status, internal identifiers, and overlap scores. The request order is fixed by the packet; do not infer a preferred request from its position. The packet is targeted, so its judgments do not estimate the frequency of semantic redundancy among all synthetic pairs.

For each language independently, compare what an expert search must find:

- **same:** both requests seek substantially the same technical capability, operation, and relevant constraints; stylistic paraphrases and minor wording differences do not create a new search intent.
- **different:** the requests require materially different evidence, operations, constraints, or stages of work. A broad request and a narrower one can be different if the extra scope would affect which experts or evidence qualify.
- **uncertain:** the requested operation, context, technical constraint, or translation cannot be interpreted reliably. Missing evidence about actual experts is not a reason to label the requests different.

Use the original need as context for terminology and scope, not as proof that either generated request is faithful. This task does not adjudicate grounding or bilingual equivalence; those require their separate fields and protocol. A high lexical overlap can still correspond to different needs, while a low overlap can conceal a paraphrase. Do not infer an expert's capability, authorship, or personal contribution from an intent-pair judgment.

Have two qualified reviewers independently complete separate copies, without seeing each other's decisions or model outputs. Record a short rationale and reviewer identifier for every labeled case; the validator rejects edited source fields and unlabeled attribution. Count missing and uncertain separately, report per-language agreement on jointly determinate cases and its denominator, and adjudicate disagreements independently. Keep all copies and mappings local under the original access terms. There are currently no human judgments.

## Public-query translation review

The combined local CoF packet contains every available Chinese translation from the two exploratory query cohorts, presented with the original English title and abstract. It hides query IDs, cohort assignment and model-screen outcomes; the content itself may reveal its source. One planned query has no translation and is not shown. Reviewers should not interpret absence as a failed translation.

Independently label each available translation as **faithful** when all substantive technical claims, quantities, negation, conditions and qualifications are preserved; **changed** when a substantive element is omitted, altered or added; or **uncertain** when the evidence or language expertise is insufficient. Preserve all source and translation fields. For every judgment record a reviewer identifier and a concise rationale identifying the affected passage or the reason for uncertainty. Two qualified reviewers should work independently on separate copies, without model judgments or each other's decisions, before adjudication.

Use `experiments.bilingual.check_cof_translation_review` to validate each copy and `experiments.bilingual.analyze_cof_translation_reviews` to summarize jointly determinate agreement. Missing and uncertain decisions remain unknown. Different identifiers do not by themselves verify independence or expertise. Translation fidelity does not establish reviewer relevance, source entailment or industrial expert fit. There are currently no completed human judgments.

For the private development-only joint dense/lexical review packet, use two independent copies of `blind-review.jsonl`. Each candidate view has three separate fields: `method_fit` (whether the displayed evidence supports a relevant technical method), `constraint_support` (whether the displayed evidence supports the stated requirement or constraint), and `personal_contribution` (whether it supports a specific individual's role). Use `supported`, `not_supported_by_view`, or `uncertain`; leave unreviewed fields null. `not_supported_by_view` describes only the displayed source scope, never a judgment that the person lacks capability. Enter a reviewer ID and rationale for any completed field; do not infer an individual's contribution from coauthorship alone. The packet hides method order, but source text can reveal identities. Validate each copy using `PYTHONPATH=src:. python -m experiments.retrieval.check_joint_review <packet-directory> <review-file>`. The validator checks frozen form and mapping hashes, unchanged source views, permitted labels and reviewer attribution; it does not validate the factual content of judgments. The packet and completed reviews contain private text and stay local.
