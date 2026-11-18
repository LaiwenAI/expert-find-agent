# PLOS CRediT Role Pilot

Data provided by PLOS. Each original article and its authors are credited through the DOI links in [the manifest](manifest.json). All 14 source XML records declare [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This adaptation selects and normalizes the structured author-role declarations. The [CRediT role vocabulary](https://credit.niso.org/) is maintained by NISO.

The pilot uses 14 PLOS ONE articles returned by a frozen title query about machine translation, information retrieval, and recommendation systems. Twelve articles contribute 132 paper-and-role queries. Each query contrasts authors with a declared role against coauthors without that declaration. There are 244 positive author-role labels and 385 same-paper role absences.

[queries.json](queries.json) uses one-based JATS author ordinals. The DOI and source field identify the attributable record. An undeclared role leaves the author's experience and ability unknown. These samples evaluate published contributor attribution on known papers. Enterprise suitability and independent adjudication require separate evidence.

Rebuild with `python -m experiments.credit.plos_role_inventory --output <new-directory>`. The live query can change as publisher data change. The manifest pins the captured search result and each XML digest. To rebuild from that exact snapshot, pass `--source-cache <snapshot-directory>` containing `search.json` and the corresponding XML files. The released fixture includes the derived labels and the manifest.
