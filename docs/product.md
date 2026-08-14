# Product

## Product Vision

ClipAI becomes a personal AI editor that learns what makes each streamer distinctive.

It is not a generic clipping service and it is not intended to replace a human editor.

## Target User

Initial target:

- YouTube streamer,
- approximately 3–10 concurrent viewers,
- no dedicated clip editor,
- produces long-form livestreams,
- lacks time to review complete archives.

## Core User Problem

Interesting moments exist but are buried in hours of footage.

The product must reduce review effort before attempting automated editing.

## MVP User Flow

1. The user provides a YouTube channel URL.
2. ClipAI selects historical channel content for initial analysis.
3. ClipAI builds versioned knowledge about the streamer.
4. The user submits or selects a new stream.
5. ClipAI generates ranked, variable-length clip candidates.
6. The user rates each candidate.
7. Future rankings use that feedback.

## Initial Learning Range

Use:

- the latest approximately 50 hours of available content,
- plus up to 10 representative videos.

Representative videos may include the channel's most viewed, most commented, or manually selected videos.

The selection policy must remain configurable.

## Candidate Definition

A candidate:

- has an AI-selected start and end,
- is between 15 seconds and a hard maximum of 15 minutes,
- contains a complete setup and payoff when possible,
- includes category scores,
- includes an overall score,
- includes human-readable selection reasons.

Natural Scene completion takes priority over compactness. Fifteen minutes is a safety
limit, not a target: a Scene that completes in 30 seconds should not be padded, while a
multi-minute setup, climax, reaction, and aftermath must not be cut at the former
120-second limit.

## Highlight Categories

Maintain separate scores rather than only one opaque score.

Initial categories:

- humor,
- great play,
- emotional moment,
- memorable quote,
- strong reaction,
- story or payoff,
- viewer interaction,
- callback or running joke.

An overall rank may combine these dimensions, but the component scores must remain visible.

## Feedback

The minimum feedback interface is:

- ◎ Excellent
- ○ Usable
- × Reject

Feedback may include one or more reason tags:

- humor,
- great play,
- emotional,
- quote,
- reaction,
- story,
- viewer interaction,
- callback,
- other.

Free-text notes are optional, not required.

## Streamer Context Scope

The streamer model may learn context such as:

- game or content genre,
- stream category,
- solo or collaboration,
- recurring collaborators,
- recurring audience interactions.

The MVP does not model external factors such as weather, season, or day of week unless later evidence shows material value.

## MVP Non-Goals

- automatic Shorts production,
- automatic publishing to social platforms,
- real-time clipping,
- Twitch or Kick support,
- cloud SaaS deployment,
- fully autonomous editorial decisions,
- full-video multimodal reasoning by an expensive LLM.

## Success Criteria

The MVP is successful when it reliably reduces review time and improves its candidate ranking from repeated human feedback.
