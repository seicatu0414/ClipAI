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
