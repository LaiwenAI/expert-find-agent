# Enterprise review readiness

The local fresh-enterprise review file contains 700 candidate views: 560 development and 140 held-out. All annotation fields are null, including grade, evidence citations, mechanism, assumptions, ownership verification and annotator. The exported review rows have no explicit silver-grade field. This check does not establish that reviewers are blinded to candidate-generation methods or that text itself cannot reveal source identity.

The separate judgment file records 183 model-silver grades, 377 unjudged development views and 140 unjudged held-out views. None of the 700 review rows contains a completed human annotation. The candidate pool was constructed using dense retrieval and random views and is not an exhaustive or unbiased pool. Human judgments must be independently collected, source ownership checked and missing labels retained before reporting expert-ranking accuracy. The held-out cases must remain unavailable for iterative method tuning.
