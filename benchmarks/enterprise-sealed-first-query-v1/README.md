# Sealed first-query evaluation of role-aware search

Prespecification for 123 metadata-frozen enterprise groups, one original
request per group. Use the same frozen 52,667-source index, checkpoints,
query interpretation, five candidate slots, and verifier throughout.
No participant feedback or test-driven updates are used. The panel's
initial source and organizational overlap screens are documented in the
[panel audit](../enterprise-blind-panel-v1). They do not establish
independence for all organizations or near-duplicate needs.

The primary comparison holds the initial dense shortlist fixed and gives
both methods the same two-passage rescue cap on initial abstention. The
candidate method searches the entire personal-role passage index. The
comparator searches that index only for experts in the initial shortlist.
The dense route without rescue and the same three strategies starting
from a hybrid shortlist are secondary comparisons. All methods use the
same frozen relevance reader. Realized rescue work can be unequal.
Report per-arm calls and checks. Comparisons restricted to equal realized
cost are descriptive post hoc analyses. Rotate arm order across queries.

## Endpoints

- A supported slot requires independent evidence for relevance, every
  applicable prior qualification, and personal contribution. Denominator
  is five slots per frozen request. Abstentions and empty slots remain
  in the denominator. A displayed view binds the exact source spans
  to the original request and expert. Method and rank stay blinded.
- A result supports a practically meaningful quality gain when the
  paired 95% lower limit for the primary contrast exceeds 2 percentage
  points of supported P@5. Pair the bootstrap by frozen group and
  preserve the common judgment of an identical evidence view. Retain
  finite-panel lower and upper bounds when labels remain unresolved.
- The separate simultaneous absolute gate uses a 60% group coverage
  lower bound and a 5% upper bound on adverse risk conditional on issuance.
  A group is adverse if an emitted recommendation fails any required
  field. Treat unresolved issued groups conservatively as adverse for
  the upper bound. The panel may be underpowered for this gate.
- Record serial P95 for parsing, lexical and dense retrieval, verification
  and fallback, and report offline index build cost. Compare latency
  after a 1-point quality noninferiority gate, with both risk and
  coverage gates met. This execution measures warm local serial runtime.
  A separate sustained-load run is needed for queueing, throughput, and
  service cost per successful query.

## Integrity

Freeze the protocol and execution code before materializing query text.
Pin original text and group hashes, source, vectors and role index,
model revisions and the accepted runner. The builder validates both
original quotations and proposed source ownership before creating the
two independent review forms. Reviewers may flag incorrect or missing
parsed qualifications. Adjudicate divergence with a third reviewer.
Keep the demand text, arm outcomes and reviewer records private. Share
aggregate findings after the blind review is reconciled.

## Score reconciled reviews

Run the existing joint-review reconciler on the two private forms and,
when necessary, an independent adjudication file. Then apply the paired
scorer. It rejects missing reviews and outstanding disagreements.
Explicitly uncertain judgments remain unresolved and enter the interval
bounds. Distinct reviewer files and identifiers are integrity checks;
reviewer qualifications and independence must be established by the
study team.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.reconcile_joint_review \
  PRIVATE_PACKET REVIEW_A REVIEW_B --adjudication REVIEW_C --output PRIVATE_RECONCILIATION
PYTHONPATH=src:. python -m experiments.retrieval.score_sealed_first_query \
  PROTOCOL LOCK FRAME PRIVATE_PACKET PRIVATE_RECONCILIATION PRIVATE_AGGREGATE \
  PRIVATE_TRACE NEW_AGGREGATE_REPORT
```

## Offline reviewer interface

Open [offline-review.html](offline-review.html) locally in a browser.
Load one assigned reviewer JSONL form, enter your reviewer ID, complete
the three judgments and rationale for each case, and export an annotated
copy. To resume, load your exported copy. Export before closing the
page. The interface reads only the selected file, shows no arm predictions,
and makes no network requests. Run `check_joint_review` on the exports
before reconciliation. Only the generic interface is tracked here.
