ALTER TABLE transcripts
    ADD COLUMN IF NOT EXISTS audio_artifact_path text;

DO $$ BEGIN
    CREATE TYPE event_job_status AS ENUM ('pending', 'running', 'completed', 'failed');
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

CREATE TABLE IF NOT EXISTS event_detection_jobs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    transcript_id uuid NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
    status event_job_status NOT NULL DEFAULT 'pending',
    progress smallint NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
    error text,
    detector_version text NOT NULL,
    configuration jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    completed_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS event_detection_jobs_claim_idx
    ON event_detection_jobs (created_at)
    WHERE status = 'pending';

CREATE TABLE IF NOT EXISTS events (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    event_detection_job_id uuid NOT NULL
        REFERENCES event_detection_jobs(id) ON DELETE CASCADE,
    transcript_id uuid NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
    event_type text NOT NULL,
    start_seconds double precision NOT NULL CHECK (start_seconds >= 0),
    end_seconds double precision NOT NULL CHECK (end_seconds >= start_seconds),
    confidence double precision NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    source_signals jsonb NOT NULL,
    explanation text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS events_timeline_idx
    ON events (transcript_id, start_seconds, end_seconds);
