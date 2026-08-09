DO $$ BEGIN
    CREATE TYPE job_status AS ENUM ('pending', 'running', 'completed', 'failed');
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE source_kind AS ENUM ('local_file', 'youtube');
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

CREATE TABLE IF NOT EXISTS transcription_jobs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_kind source_kind NOT NULL,
    source text NOT NULL,
    status job_status NOT NULL DEFAULT 'pending',
    progress smallint NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
    error text,
    model_size text NOT NULL,
    language text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    completed_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS transcription_jobs_claim_idx
    ON transcription_jobs (created_at)
    WHERE status = 'pending';

CREATE TABLE IF NOT EXISTS transcripts (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id uuid NOT NULL UNIQUE REFERENCES transcription_jobs(id) ON DELETE CASCADE,
    source_duration_seconds double precision,
    detected_language text,
    detected_language_probability double precision,
    model_size text NOT NULL,
    device text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS transcript_segments (
    id bigserial PRIMARY KEY,
    transcript_id uuid NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
    segment_index integer NOT NULL,
    start_seconds double precision NOT NULL CHECK (start_seconds >= 0),
    end_seconds double precision NOT NULL CHECK (end_seconds >= start_seconds),
    text text NOT NULL,
    average_log_probability double precision,
    no_speech_probability double precision,
    UNIQUE (transcript_id, segment_index)
);

CREATE INDEX IF NOT EXISTS transcript_segments_timeline_idx
    ON transcript_segments (transcript_id, segment_index);
