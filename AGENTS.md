# ClipAI Agent Rules

## Purpose

This file defines the global rules for AI agents working on ClipAI.
It is intentionally short because it may be loaded for every task.

## Mission

ClipAI helps small streamers discover valuable moments buried in long archives.

The product must optimize for:

> moments that are meaningful or entertaining for this specific streamer

It must not optimize only for generic virality.

The human remains the final editor and decision-maker.

## Required Reading Order

1. Read this file.
2. Read `README.md`.
3. Read only the task-relevant file under `docs/`.
4. Stop reading when enough context has been obtained.

Do not load the entire documentation set by default.

## Context Budget

Context is a limited engineering resource.

- Do not create a new document unless an existing document cannot own the knowledge.
- Do not duplicate the same rule in multiple files.
- Keep one responsibility per document.
- Prefer links over repeated explanations.
- When a document becomes too broad, propose a split before making it larger.
- English and Japanese files must remain semantically equivalent.

## Decision Authority

Technical decisions that do not change product intent may be made by the acting architect.

Ask the product owner only when a decision materially changes:

- target users,
- supported platforms,
- user experience,
- product scope,
- cost model,
- privacy or legal policy,
- business priority.

Do not ask the product owner to choose between ordinary implementation details.

## Architecture Rules

- Use a local-first architecture for the initial version.
- Keep API work and heavy AI processing separated.
- Keep pipeline stages replaceable.
- Keep prompts outside application source code.
- Keep business rules independent from specific AI providers.
- Use LLMs for reasoning after candidate reduction, not for brute-force full-video analysis.
- Preserve analysis history when experiments need comparison.
- Every AI-generated recommendation must include reasons and evidence.

## Change Policy

If an existing decision appears wrong:

1. identify the concrete problem,
2. propose the smallest correction,
3. explain the trade-off,
4. do not silently change product intent.

Do not create process, governance, or documentation for its own sake.

## Implementation Priority

When several valid implementations exist, prefer:

1. simplicity,
2. maintainability,
3. explainability,
4. extensibility,
5. performance.

Avoid premature optimization.

## Definition of Done

A task is done when:

- the requested behavior is complete,
- relevant tests pass,
- responsibilities remain separated,
- changed decisions are reflected in the relevant document,
- English and Japanese documentation remain aligned,
- no unnecessary files or abstractions were added.
