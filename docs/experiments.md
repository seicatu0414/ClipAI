# Experiments and Evaluation

## End Boundary Comparison

Compare the same Event and transcript with the previous and current pipeline. Record old
and new end timestamps, delta seconds, inferred TopicWindow, selection reason, confidence,
and source signals. Review content, start boundary, and end boundary separately so future
feedback learning can distinguish these failure modes.

## Purpose

ClipAI contains uncertain AI behavior.
Changes must be evaluated rather than accepted because they sound plausible.

## Core Hypotheses

### H1 — Personalization

StreamerKnowledge improves candidate acceptance compared with a generic highlight detector.

### H2 — Candidate Reduction

Cheap audio, transcript, and metadata signals can reduce LLM workload without unacceptable loss of strong moments.

### H3 — Scene-Aware Variable Duration

Scene-aware windows between 15 seconds and a 15-minute hard maximum preserve causal
setup, payoff, reaction, and aftermath better than fixed 30/60-second windows or the old
120-second cap, without padding already-complete short Scenes.

### H4 — Feedback

◎ / ○ / × feedback with reason tags improves later ranking.

### H5 — Vision Value

Vision adds enough incremental accuracy to justify its processing cost only after the audio and transcript baseline is established.

## Evaluation Dataset

Start with a small private benchmark:

- several streams from one target creator,
- later add a small number of additional creators,
- preserve the original archives,
- maintain a human-reviewed list of strong and weak regions.

Do not require professional editors.
The project owner may perform review because the goal is to reduce repeated archive review, not eliminate all human judgment.

## Metrics

Track at minimum:

- candidate acceptance rate,
- Precision@20,
- Precision@30,
- duplicate or overlapping candidate rate,
- coverage of known strong moments,
- false-positive categories,
- average review duration,
- processing time per video hour,
- GPU memory usage,
- LLM tokens or calls,
- human-rated explanation quality.

Recall can be estimated only when the review set contains known missed moments.

## Experiment Record

Each experiment must record:

- hypothesis,
- dataset version,
- pipeline version,
- model and prompt versions,
- configuration,
- output metrics,
- qualitative failures,
- decision.

## Acceptance Rule

An AI change should not become the default unless it:

- improves a target metric,
- fixes a documented failure mode,
- or materially reduces processing cost without unacceptable quality loss.

Keep failed experiments; they prevent repeated dead ends.

## v0.4 Candidate Reduction Baseline

For a representative four-hour stream, record candidate count, duration distribution,
pairwise overlap rate, and category distribution from the candidate-job results. The
baseline succeeds when it yields 20–30 reviewable candidates, every duration is between
15 seconds and the 15-minute hard maximum, explanations cite stored Scene/thread evidence,
and no pair reaches the configured overlap
threshold. Compare by candidate job ID; pinned inputs and analysis-version metadata make
each run reproducible. Human usefulness remains the deciding measure.

## Scene Boundary Comparison

For the same transcript and anchor Event, record old/new start and end, duration,
TopicWindow, SceneWindow and phase, open/resolved threads, selected candidate, boundary
confidence, Scene completion confidence, detailed-analysis usage, and reason. Include
short, 2–5 minute, approximately 10-minute, and hard-limit examples. A change is accepted
only when human review finds the causal Scene more complete without unnecessary following
Scenes; low-confidence cases must show whether bounded context expansion improved certainty.

## v0.5 Feedback Comparison

Review candidates, then call `GET /v1/streamers/{streamer_id}/preferences/compare` with
`before_version_id` and `after_version_id`. Record accepted average rank and
Precision@20/30 on the same reviewed set. Promote a newer preference only when acceptance
ranking improves without a qualitative regression. Notes and `other` are stored but do
not affect weights in this baseline.
