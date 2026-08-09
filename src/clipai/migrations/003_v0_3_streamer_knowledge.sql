CREATE TABLE IF NOT EXISTS streamers (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    youtube_channel_id text UNIQUE,
    channel_url text NOT NULL,
    display_name text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS streams (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    streamer_id uuid NOT NULL REFERENCES streamers(id) ON DELETE CASCADE,
    transcript_id uuid NOT NULL UNIQUE REFERENCES transcripts(id) ON DELETE CASCADE,
    youtube_video_id text,
    source_url text,
    title text NOT NULL,
    published_at timestamptz NOT NULL,
    duration_seconds double precision NOT NULL CHECK (duration_seconds > 0),
    view_count bigint NOT NULL DEFAULT 0 CHECK (view_count >= 0),
    comment_count bigint NOT NULL DEFAULT 0 CHECK (comment_count >= 0),
    manually_selected boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS streams_history_selection_idx
    ON streams (streamer_id, published_at DESC);

DO $$ BEGIN
    CREATE TYPE knowledge_job_status AS ENUM ('pending', 'running', 'completed', 'failed');
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

CREATE TABLE IF NOT EXISTS knowledge_jobs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    streamer_id uuid NOT NULL REFERENCES streamers(id) ON DELETE CASCADE,
    status knowledge_job_status NOT NULL DEFAULT 'pending',
    progress smallint NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
    error text,
    provider text NOT NULL,
    model text NOT NULL,
    prompt_version text NOT NULL,
    configuration jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    completed_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS streamer_knowledge_versions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    streamer_id uuid NOT NULL REFERENCES streamers(id) ON DELETE CASCADE,
    knowledge_job_id uuid NOT NULL UNIQUE REFERENCES knowledge_jobs(id) ON DELETE CASCADE,
    version_number integer NOT NULL,
    previous_version_id uuid REFERENCES streamer_knowledge_versions(id),
    provider text NOT NULL,
    model text NOT NULL,
    prompt_version text NOT NULL,
    configuration jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (streamer_id, version_number)
);

CREATE TABLE IF NOT EXISTS knowledge_observations (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    knowledge_version_id uuid NOT NULL
        REFERENCES streamer_knowledge_versions(id) ON DELETE CASCADE,
    category text NOT NULL,
    statement text NOT NULL,
    origin text NOT NULL CHECK (origin IN ('observed', 'inferred')),
    confidence double precision NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS knowledge_evidence (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    observation_id uuid NOT NULL REFERENCES knowledge_observations(id) ON DELETE CASCADE,
    transcript_id uuid NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
    segment_index integer NOT NULL,
    start_seconds double precision NOT NULL,
    end_seconds double precision NOT NULL,
    quote text NOT NULL
);

CREATE INDEX IF NOT EXISTS knowledge_observations_version_idx
    ON knowledge_observations (knowledge_version_id, category);
