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
