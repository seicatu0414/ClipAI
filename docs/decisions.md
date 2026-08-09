# Accepted Decisions

This file records only durable decisions that materially guide future work.
Do not use it as a diary.

## Product

- Initial platform: YouTube only.
- Analyze livestream archives and ordinary uploaded videos.
- Initial users: small streamers with roughly 3–10 concurrent viewers.
- Human performs final selection and editing.
- Initial learning range: latest approximately 50 hours plus up to 10 representative videos.
- Candidate duration: AI-selected, normally 15–120 seconds.
- Scores: multidimensional categories plus an overall rank.
- Feedback: ◎ / ○ / × with optional reason tags.
- Vision: deferred until audio and transcript analysis is validated.

## Architecture

- Initial deployment: local only.
- Host environment: Windows 11 with RTX 4070 Ti.
- Use Docker Compose for reproducibility.
- Separate API responsibilities from heavy Worker processing.
- Preferred API: FastAPI.
- Preferred database: PostgreSQL with pgvector.
- Preferred transcription: faster-whisper.
- Preferred local LLM runtime: Ollama.
- LLM provider must be replaceable.
- Do not send complete videos to an LLM.
- Use deterministic or lightweight filtering before LLM reasoning.
- Use a simple database-backed job mechanism before adopting separate queue infrastructure.
- Store StreamerKnowledge as versioned, evidence-backed knowledge.

## Documentation

- English is the AI-facing source of truth.
- Japanese is a same-tree semantic mirror for owner review.
- AI agents receive only the English tree when possible.
- Global context consists of `AGENTS.md` and `README.md`.
- Other documents are read only when relevant to the task.
- New documents require a clear responsibility and must not duplicate existing knowledge.
