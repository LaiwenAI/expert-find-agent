# Transfer Constraint Counterfactuals

This synthetic diagnostic covers 48 base cases across 12 engineering and operations settings. Each base case has two requests with identical evidence and a single changed bound. The comparable measurement meets one bound and violates the other. A third case hides that measurement while retaining a proposed target, a measurement from another setting, and an unrelated observation.

The decisions describe eligibility under the specific displayed constraint. They do not establish a person's expertise or a method's performance in a real deployment.

- **Eligible** means the comparable measurement meets the bound and a conditional recommendation remains available.
- **Reject** means the comparable measurement violates the bound.
- **Unknown** means there is no comparable measurement in the supplied cards.

`query`, `scope`, and each card's `text` contain model-facing inputs. The source-selection experiment does not expose the bound to the selector. The calculator consumes the selected card's pre-normalized `reading`, the comparator, and the bound. Card roles, `gold_source`, and `gold` are evaluation metadata. The verified-input gate uses the gold source as a diagnostic upper bound.

The generator and GPU evaluation live in the [counterfactual experiment](../../experiments/transfer/constraint_counterfactual.py). Scores include the number of pairs for which both intact sides are correct and the number of eligibility decisions made after the comparable measurement was hidden. Do not train on individual members and test on other members of the same base case.

The dataset consists of original fictional fixtures. It includes no enterprise requests, researcher identities, or copied source documents.
