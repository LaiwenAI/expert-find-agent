"""Change only the source selection policy of frozen dense reviewer search."""

from experiments.retrieval.cof_live_certified_batching import TOP
from experiments.retrieval.cof_merit_resident_bundles import ResidentIndex, WIDTH, arguments, run

ARMS = ("dense", "bundle_only", "bundle_mixed")


class FixedNominationIndex(ResidentIndex):
    def __init__(self, reference, papers, model, tokenizer, template):
        super().__init__(reference, papers, model, tokenizer, template)
        self.owner_positions = {owner: i for i, owner in enumerate(reference.owner_ids)}

    def search_query(self, query, embedding, arm):
        hits = self.compact.search_query(query, embedding, "dense")
        if arm == "dense":
            return hits

        _, source_bundles = self.merit_scores(query, "merit_multi")
        chosen = []
        for owner, dense in hits[:TOP]:
            bundle = source_bundles[self.owner_positions[owner]]
            if arm == "bundle_only":
                preference = (*bundle, *dense)
            else:
                # Keep the first dense source and give the bundle one slot
                # before filling the remaining capacity from dense order.
                addition = [doc for doc in bundle if doc != dense[0]]
                preference = (dense[0], *addition[:1], *dense, *bundle)
            chosen.append((owner, list(dict.fromkeys(preference))[:WIDTH]))
        return [*chosen, *hits[TOP:]]


if __name__ == "__main__":
    run(arguments(__doc__), index_type=FixedNominationIndex, arms=ARMS, entrypoint=__file__)
