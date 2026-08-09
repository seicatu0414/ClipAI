# ClipAI Context Pack

ClipAI is a local-first research and development project for a personal AI editing assistant.

## Problem

Small streamers, especially those with roughly 3–10 concurrent viewers, often have:

- no dedicated clip editor,
- long archives they cannot review,
- entertaining moments that remain undiscovered.

## Product Goal

Reduce a long archive into a manageable set of ranked highlight candidates that a human can review and edit.

A representative success target is:

> reduce a four-hour stream to approximately 20–30 useful candidates

## Current Scope

- Platform: YouTube
- Inputs: livestream archives and ordinary uploaded videos
- Output: ranked, explainable clip candidates
- Final editing: human
- Development environment: local machine
- Target GPU: NVIDIA RTX 4070 Ti

## Knowledge Map

Read only the document relevant to the task:

- [`docs/product.md`](docs/product.md): users, scope, requirements, non-goals
- [`docs/architecture.md`](docs/architecture.md): system boundaries and AI pipeline
- [`docs/domain.md`](docs/domain.md): core concepts and ownership
- [`docs/roadmap.md`](docs/roadmap.md): phased delivery order
- [`docs/experiments.md`](docs/experiments.md): hypotheses, datasets, and evaluation
- [`docs/decisions.md`](docs/decisions.md): accepted product and technical decisions

## Documentation Policy

The English tree is the source of truth for AI agents.

The Japanese tree mirrors the same structure and meaning for owner review.

A design change is incomplete until both versions are updated.

## Local Development

Copy `.env.example` to `.env`, then start the foundation with:

```powershell
docker compose up --build
```

The API health endpoint is `http://localhost:8000/health`. The API and Worker run as separate processes; heavy processing belongs only in the Worker. Stop the stack with `docker compose down`.

Quality checks:

```powershell
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src tests
pytest
```

## v0.1 Local Transcription

Place local media under `data/` (mounted as `/data` in the containers), or use a public YouTube video URL. Start the CPU-safe stack with:

```powershell
docker compose up --build
```

For NVIDIA GPU acceleration, use the GPU override:

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

Submit and inspect a job through the API:

```powershell
$job = Invoke-RestMethod -Method Post -Uri http://localhost:8000/v1/transcription-jobs `
  -ContentType application/json -Body '{"source":"/data/example.mp4"}'
Invoke-RestMethod http://localhost:8000/v1/transcription-jobs/$($job.id)
```

After completion, inspect ordered segments without loading the full transcript:

```powershell
Invoke-RestMethod "http://localhost:8000/v1/transcripts/$($job.transcript_id)/segments?offset=0&limit=100"
```

YouTube smoke test:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/v1/transcription-jobs `
  -ContentType application/json -Body '{"source":"https://www.youtube.com/watch?v=VIDEO_ID"}'
```

## v0.2 Basic Event Detection

Run lightweight event detection for one completed transcript:

```powershell
$eventJob = Invoke-RestMethod -Method Post -Uri http://localhost:8000/v1/event-detection-jobs `
  -ContentType application/json -Body (ConvertTo-Json @{ transcript_id = $job.transcript_id })
Invoke-RestMethod http://localhost:8000/v1/event-detection-jobs/$($eventJob.id)
Invoke-RestMethod http://localhost:8000/v1/transcripts/$($job.transcript_id)/events
```

The timeline contains event type, start/end time, confidence, source signals, and an explanation. Detection uses RMS loudness/silence features and versioned Japanese transcript rules. Thresholds are configurable through the `CLIPAI_EVENT_*` environment variables. Events are evidence regions, not ranked clip candidates.

## v0.3 StreamerKnowledge

Pull the default local model once:

```powershell
docker compose exec ollama ollama pull qwen2.5:7b-instruct
```

Create a Streamer with `POST /v1/streamers`, then register completed transcripts and their historical metadata with `POST /v1/streams`. Start knowledge generation with `POST /v1/knowledge-jobs` using the Streamer ID and poll `GET /v1/knowledge-jobs/{job_id}`.

Inspect the current evidence-backed version at:

```text
GET /v1/streamers/{streamer_id}/knowledge/current
```

The default selection uses the latest approximately 50 hours plus up to 10 representative registered streams. Each transcript is processed in bounded chunks; the complete history is never placed in one prompt. The default provider/model/prompt are `ollama`, `qwen2.5:7b-instruct`, and `prompts/streamer_knowledge/v1.md`. Every observation includes confidence, `observed` or `inferred` origin, and timestamped transcript evidence.

Example observation:

```json
{
  "category": "recurring_phrase",
  "statement": "成功時に『やった』と繰り返す",
  "origin": "observed",
  "confidence": 0.82,
  "evidence": [{"segment_index": 42, "start_seconds": 315.2, "quote": "やった！"}]
}
```

## v0.4 Personalized Clip Candidates

Create a candidate job after event detection and StreamerKnowledge are complete:

```powershell
$candidateJob = Invoke-RestMethod -Method Post -Uri http://localhost:8000/v1/candidate-jobs `
  -ContentType application/json -Body (ConvertTo-Json @{
    streamer_id = $streamer.id
    transcript_id = $job.transcript_id
  })
Invoke-RestMethod http://localhost:8000/v1/candidate-jobs/$($candidateJob.id)
Invoke-RestMethod http://localhost:8000/v1/candidate-jobs/$($candidateJob.id)/candidates
```

The worker constructs and deduplicates 15–120 second windows from inexpensive event
signals, targeting 25 by default. Only the reduced set is sent to the LLM with relevant
evidence-backed StreamerKnowledge. Results include eight category scores, overall rank,
confidence, reasons, event IDs, and exact analysis-version metadata.

## v0.5 Feedback Learning

Submit ◎, ○, or × to `POST /v1/candidates/{candidate_id}/feedback` with optional
reason tags and a note. Each response creates a separate feedback record and an immutable
streamer preference version. Future candidate jobs pin the current version and apply its
transparent eight-category weights over the unchanged global score composition.

List versions with `GET /v1/streamers/{streamer_id}/preferences`, roll back by appending
a version with `POST /v1/streamers/{streamer_id}/preferences/rollback`, and compare two
versions with `GET /v1/streamers/{streamer_id}/preferences/compare`. For example, an ◎
on a humor-heavy candidate tagged `humor` changes that weight from 1.0000 to 1.0810.
Notes and `other` tags are retained but do not alter weights in v0.5.
