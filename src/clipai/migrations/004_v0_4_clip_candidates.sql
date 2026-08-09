DO $$ BEGIN
    CREATE TYPE candidate_job_status AS ENUM ('pending', 'running', 'completed', 'failed');
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

CREATE TABLE IF NOT EXISTS candidate_jobs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    streamer_id uuid NOT NULL REFERENCES streamers(id) ON DELETE CASCADE,
    transcript_id uuid NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
    event_detection_job_id uuid NOT NULL REFERENCES event_detection_jobs(id),
    knowledge_version_id uuid NOT NULL REFERENCES streamer_knowledge_versions(id),
    status candidate_job_status NOT NULL DEFAULT 'pending',
    progress smallint NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
    error text,
    pipeline_version text NOT NULL,
    provider text NOT NULL,
    model text NOT NULL,
    prompt_version text NOT NULL,
    configuration jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    completed_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS clip_candidates (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_job_id uuid NOT NULL REFERENCES candidate_jobs(id) ON DELETE CASCADE,
    transcript_id uuid NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
    rank integer NOT NULL CHECK (rank > 0),
    start_seconds double precision NOT NULL CHECK (start_seconds >= 0),
    end_seconds double precision NOT NULL CHECK (end_seconds > start_seconds),
    category_scores jsonb NOT NULL,
    overall_score double precision NOT NULL CHECK (overall_score BETWEEN 0 AND 1),
    confidence double precision NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    reasons jsonb NOT NULL,
    event_ids jsonb NOT NULL,
    knowledge_observation_ids jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (candidate_job_id, rank)
);

CREATE INDEX IF NOT EXISTS clip_candidates_job_rank_idx
    ON clip_candidates (candidate_job_id, rank);
