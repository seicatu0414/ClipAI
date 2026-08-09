# Architecture

## Initial Environment

- Windows 11 host
- NVIDIA RTX 4070 Ti
- Docker Compose
- Local storage
- No public deployment

## Main Components

### API

Preferred implementation: FastAPI.

Responsibilities:

- accept channel and video requests,
- expose job state,
- expose analysis results,
- accept human feedback.

The API must not perform heavy video or AI processing.

### Worker

Responsibilities:

- acquire source media,
- extract audio,
- run transcription,
- extract features,
- detect candidate events,
- build or update StreamerKnowledge,
- rank ClipCandidates.

Heavy processing belongs here.

### Database

Preferred implementation:

- PostgreSQL
- pgvector extension

Responsibilities:

- metadata,
- analysis history,
- transcripts and segments,
- detected events,
- candidate scores and reasons,
- human feedback,
- versioned StreamerKnowledge,
- embeddings where semantic retrieval is justified.

Large video and audio assets should remain on local storage rather than inside PostgreSQL.

### Local LLM

Preferred default: Ollama.

The application must depend on an LLM provider interface rather than Ollama-specific business logic.

External providers may be added for comparison, but they are not required for the first working version.

## AI Pipeline

1. Resolve YouTube channel or video metadata.
2. Acquire the selected video or audio.
3. Extract normalized audio with FFmpeg.
4. Generate a timestamped transcript with faster-whisper.
5. Extract inexpensive deterministic features.
6. Detect potentially meaningful event regions.
7. Merge nearby events into candidate windows.
8. Retrieve only relevant StreamerKnowledge.
9. Use an LLM to reason about reduced candidate windows.
10. Produce multidimensional scores and explanations.
11. Present candidates to the human.
12. Store feedback and update future ranking behavior.

## Candidate Reduction Principle

Do not send the entire video to an LLM.

Use cheaper signals first, such as:

- voice activity,
- loudness change,
- speech rate change,
- laughter or vocal reaction signals,
- transcript semantics,
- unusual silence,
- repeated phrases,
- confidence or contradiction language,
- chat or comment evidence when available.

Vision analysis is deferred until the audio and transcript pipeline has demonstrated value.

## Replaceable Boundaries

The following must be replaceable behind interfaces:

- video provider,
- transcription engine,
- feature extractor,
- event detector,
- LLM provider,
- embedding provider,
- candidate scorer,
- knowledge updater.

## Data History

Do not overwrite experiment results that are needed for comparison.

A new pipeline, model, prompt, or configuration should be identifiable in stored analysis metadata.

StreamerKnowledge must be versioned so changes can be inspected and rolled back.

Streamer-specific preferences are a versioned layer over the global candidate scorer.
A candidate job pins one preference version. Feedback appends a feedback record and a new
preference version; it never updates stored candidates. Rollback is also append-only.

## Initial Job Processing

A simple database-backed job mechanism is sufficient for the local MVP.

Do not introduce Redis, RabbitMQ, Celery, or distributed orchestration until local workload proves the need.

## Security and Privacy

- Never commit tokens, cookies, downloaded private content, or personal datasets.
- Keep secrets in local environment variables.
- Analyze only content the user is authorized to access.
- Treat personality descriptions as probabilistic observations, not objective facts.
