# Transfer across topics and governing clause layouts

Construct 168 bilingual controls from public procedural topics
and authored scopes. Within a four-view group, preserve the
target excerpt exactly. Change the actor, timing and binding
language so it denotes applicant prior qualification, future
delivery, an existing operating condition, or another
party's background. The training set has 96 views. The
calibration set has 24 views and a new layout and topics.
The 48 held views use different topics and two more layouts.
One held layout puts an unrelated prior-qualification cue
earlier in the request. The other binds the exact target from
a later sentence. In both, the immediately adjacent text is
constant across the four roles.

Fit a ridge head over frozen Qwen3-Embedding-0.6B features.
Compare the target only, a local window with an optional
preceding heading, and the full request split around the
exact target. With full context, also test training on the
same rows plus positive-minus-negative differences between
views in the same group. The differences are training
examples only. Every test route sees one view at a time.
Run Qwen3-4B-Instruct-2507 with the existing frozen role
prompt on the same held target spans. Treat only its
prior-qualification choice as positive.

For each ridge head, freeze an abstention threshold from
the largest absolute score among incorrect calibration
views. Abstention is a diagnostic. Report binary qualification
results and the count of four-view groups classified without
an error.

## Results

Both Slurm tasks in array `45219` reproduce the controls,
qualification counts and confidence cutoffs.

| Route | Qualifications found / 12 | False qualifications / 36 | Complete groups / 12 | Held views accepted by confidence threshold |
| --- | ---: | ---: | ---: | ---: |
| Target only, 0.6B | 0 | 0 | 0 | 0 |
| Local, 0.6B | 0 | 0 | 0 | 0 |
| Full request, 0.6B | 8 | 0 | 8 | 0 |
| Full with paired training, 0.6B | 7 | 0 | 7 | 0 |
| Local, frozen 4B | 0 | 0 | 0 | — |
| Full request, frozen 4B | 0 | 0 | 0 | — |

The full-context head extracts some signal from the governing
text in the constructed transfer cases. Adding paired feature
differences does not improve recall here. None of the calibrated
heads makes a high-confidence decision on the held set, so the
study provides no support for skipping the full qualification
review. It measures authored counterfactuals, not enterprise
label accuracy or the speed of a production search system.

[Aggregate results](results.json) and [generated control views](controls.json)
are included. The checked-in JSON escapes non-ASCII characters
and decodes to the same dataset as the CLI output.

## Reproduce

Use a Slurm CUDA allocation with the pinned embedding and
reader checkpoints from the result manifest and fresh paths.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.enterprise_scope_transfer \
  EMBEDDING_CHECKPOINT READER_CHECKPOINT NEW_RESULTS NEW_CONTROLS
```
