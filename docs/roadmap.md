# Roadmap

## v0.1 — Local Transcription

Goal: convert one local or YouTube video into a timestamped transcript.

Deliverables:

- Docker-based local environment,
- FFmpeg preprocessing,
- faster-whisper on the RTX 4070 Ti,
- persisted transcript and segments,
- reproducible command or API entry point.

Exit condition:

A multi-hour Japanese stream can be transcribed reliably and inspected by timestamp.

## v0.2 — Basic Event Detection

Goal: find potentially interesting regions without personalizing them yet.

Deliverables:

- audio and transcript feature extraction,
- initial event taxonomy,
- event timeline,
- explainable rule or model output.

Exit condition:

The system reduces a long stream to a smaller set of review regions without using full-video LLM analysis.

## v0.3 — StreamerKnowledge

Goal: build versioned knowledge from selected historical content.

Deliverables:

- historical content selection,
- evidence-backed knowledge schema,
- configurable LLM provider,
- knowledge version history.

Exit condition:

A human familiar with the streamer considers the generated knowledge broadly recognizable and evidence-based.

## v0.4 — Personalized Clip Candidates

Goal: rank variable-length candidates using both event evidence and StreamerKnowledge.

Deliverables:

- candidate window construction,
- multidimensional category scores,
- overall ranking,
- reasons and evidence.

Exit condition:

A four-hour stream produces approximately 20–30 reviewable candidates.

Implementation baseline:

- construct 15–120 second windows from the versioned event timeline,
- reduce overlap and cap the pre-LLM set at 25 by default,
- retrieve only category-relevant knowledge from the pinned knowledge version,
- score eight clip dimensions and retain reasons plus source references,
- preserve each run's pipeline, provider, model, prompt, and configuration.

## v0.5 — Feedback Learning

Goal: use ◎ / ○ / × feedback to improve future ranking.

Deliverables:

- feedback UI or endpoint,
- reason tags,
- preference update logic,
- before-and-after evaluation.

Exit condition:

Repeated use measurably improves candidate acceptance rate.

Implementation baseline:

- store every ◎ / ○ / × separately with optional tags and note,
- append bounded, explainable category-weight versions,
- pin future candidate jobs to the current preference version,
- roll back by appending a copy of a selected historical version,
- compare accepted average rank and Precision@20/30 on the same reviewed candidates.

## Deferred

- vision analysis,
- automatic editing,
- automatic posting,
- real-time processing,
- Twitch and Kick,
- cloud deployment.
