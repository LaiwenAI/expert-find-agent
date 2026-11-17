# Aligning Experiments with the Continual Expert-Search Plan

The research plan prioritizes feedback-driven continual improvement of an expert-search agent under quality and cost constraints. Existing source ownership, evidence contracts, lexical and vector indexes, bilingual construction, and benchmark adapters remain the starting system and comparison infrastructure. Their previously reported retrieval or component results do not establish continual improvement.

## Current starting point

- The existing candidate and grounded pipelines provide frozen search, source access, and assessment paths. They do not yet provide an end-to-end feedback-driven policy update.
- The construction corpus has 1,000 real parent needs and 5,000 generated bilingual query pairs. Model screens are not independent expert-fit judgments; feedback, personal contribution, and genuine user outcomes remain unverified.
- Public reviewer ratings and EvidenceBench provide auxiliary evaluation. Reviewer ratings are not enterprise suitability labels, and model-assisted sentence aspects are not expert-level gold.
- Component timing and concurrent assessment pilots exclude parts of the agent. They are useful implementation baselines, not a full-agent quality or cost claim.

## Adjusted experimental sequence

1. **Freeze a static agent and strong retrieval baselines.** Version the corpus, source ownership, models, prompts, index, candidate and evidence budgets, failure policy, and timing scope. Include fixed lexical and dense ranking, a fixed agent, and direct feedback-free retrieval. Compare the same query cohort and report all attempts, unresolved outputs, initialization, query encoding, retrieval, reasoning, and total cost. Index changes require a separate arm.
2. **Build feedback and evaluation streams.** Keep existing construction, validation, and sealed company groups isolated. Preserve Chinese and English views of each parent together. For development, distinguish explicit corrections, exposure/choice events, and delayed outcomes. No actual user logs are presently established; synthetic feedback must be marked as simulated and assessed against independent labels when available. Do not treat unselected experts as negative or use the same model's generated feedback as gold.
3. **Compare feedback mechanisms.** Evaluate a frozen agent, a feedback-only correction or memory baseline, an adaptive-compute-only baseline, their simple combination, and a joint feedback-and-budget policy. Use the same model, snapshot, candidate pool where the causal question requires it, and feedback budget. Training-free memory/routing is the first comparison; RL, SFT or distillation remain later alternatives with their own training costs and data requirements.
4. **Test continual transfer and forgetting.** Fix update batches and order before evaluation. After each update, evaluate disjoint new tasks and a fixed old-task retention set; report quality per domain, need type, language and feedback availability. Hold the source index fixed when attributing gains to policy learning. Evaluate index updates separately. The 73-query simulation below does not satisfy this requirement.
5. **Measure quality, evidence and economics jointly.** Predefine expert-level success, source support, attribution, constraint satisfaction and unresolved outcomes using independent reviewers; treat missing information as unknown. Report P50/P95/P99 complete-query latency, token/API costs, failures, feedback, training, index and deployment amortization, and quality-cost curves. Set a meaningful non-inferiority margin before confirmation and use grouped paired uncertainty estimates; do not infer equivalence from an interval crossing zero.

## Initial feedback-route sanity check

`experiments.retrieval.cof_feedback_routing_simulation` replays previously explored CoF SIGIR rankings. In each of 30 shuffled streams, a simple epsilon-greedy selector receives only the chosen arm's reviewer-label P@5 after each query. Across overlapping 53-query post-warmup slices, mean soft P@5 is 0.412 for routing, 0.399 for fixed lexical, and 0.429 for fixed dense; dense is selected 58.7% of the time. These are simulated feedback on reused self-rated reviewer data, not user interactions or industrial labels. The shuffled streams are not independent or chronological; no real model call, latency, feedback collection, old-task retention, policy transfer, or training cost is measured. This is infrastructure exploration, not evidence of continual improvement.

## Immediate gates

- Do not claim improved enterprise precision, evidence correctness, contribution assessment, or quality-preserving speed until independent expert-level and source-level labels exist.
- Do not use held-out company groups to tune routing, feedback interpretation, stopping, or cost weights.
- Do not promote a simulated feedback policy to a user-facing default based on the replay above.
- Preserve negative and incomplete results; distinguish exposure, selected actions, missing labels, and eventual outcomes.
