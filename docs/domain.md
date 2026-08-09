# Domain

## Naming Principle

Use domain names that describe durable business meaning, not a specific AI implementation.

## Streamer

Represents the creator whose long-term patterns the product learns.

Key identity:

- internal StreamerId,
- YouTube channel ID,
- channel URL,
- display name.

A Streamer does not contain every Stream or analysis result in memory as one large object.

## Stream

Represents one source video or livestream archive.

Key information:

- StreamId,
- StreamerId,
- YouTube video ID,
- source URL,
- title,
- publication time,
- duration,
- content category metadata.

One Stream may be analyzed multiple times.

## AnalysisSession

Represents one reproducible execution of the analysis pipeline against one Stream.

It records:

- pipeline version,
- model versions,
- prompt versions,
- configuration,
- status,
- generated artifacts,
- errors and timings.

Historical sessions are retained when needed for experiment comparison.

## Transcript

Represents timestamped speech generated from one AnalysisSession.

It contains ordered segments with:

- start and end time,
- text,
- optional speaker label,
- transcription confidence metadata.

## Event

Represents a potentially meaningful occurrence detected within a Stream.

Initial event types may include:

- laughter,
- scream or loud reaction,
- singing,
- crying or emotional voice,
- unusual silence,
- victory or defeat evidence,
- memorable statement,
- viewer response,
- callback or contradiction.

An Event is evidence, not automatically a ClipCandidate.

## StreamerKnowledge

Represents versioned, evidence-backed knowledge learned about one Streamer.

It may include:

- speech patterns,
- emotional baseline,
- recurring phrases,
- recurring jokes,
- typical behavior,
- content strengths,
- collaboration patterns,
- preferred candidate categories,
- known callbacks.

Rules:

- observations include confidence and evidence,
- knowledge is versioned,
- the current version can be replaced without deleting history,
- personality language must avoid unsupported certainty.

## ClipCandidate

Represents one ranked review suggestion.

It contains:

- start and end,
- category scores,
- overall score,
- confidence,
- selection reasons,
- evidence references,
- source AnalysisSession,
- human review status.

## Feedback

Represents the human evaluation of a ClipCandidate.

It contains:

- rating,
- reason tags,
- optional note,
- timestamp.

Feedback changes future ranking preferences but does not rewrite historical candidate results.

## RankingPreference

An immutable streamer-specific set of category weights over the replaceable global ranking
logic. Each version records its predecessor, source feedback or rollback target, weights,
and explanation. Candidate analyses pin the version they used. Free-text notes are retained
but are not automatically interpreted in v0.5.

## Ownership Summary

- Streamer owns long-term identity.
- Stream belongs to one Streamer.
- AnalysisSession belongs to one Stream.
- Transcript, Event, and ClipCandidate belong to one AnalysisSession.
- Feedback belongs to one ClipCandidate.
- StreamerKnowledge belongs to one Streamer and is updated from evidence produced by AnalysisSessions.
