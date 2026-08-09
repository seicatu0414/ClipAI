CREATE TABLE IF NOT EXISTS streamer_preference_versions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    streamer_id uuid NOT NULL REFERENCES streamers(id) ON DELETE CASCADE,
    version_number integer NOT NULL,
    previous_version_id uuid REFERENCES streamer_preference_versions(id),
    source_feedback_id uuid,
    rollback_of_version_id uuid REFERENCES streamer_preference_versions(id),
    category_weights jsonb NOT NULL,
    explanation jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (streamer_id, version_number)
);

CREATE TABLE IF NOT EXISTS candidate_feedback (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_id uuid NOT NULL REFERENCES clip_candidates(id) ON DELETE CASCADE,
    streamer_id uuid NOT NULL REFERENCES streamers(id) ON DELETE CASCADE,
    rating text NOT NULL CHECK (rating IN ('excellent', 'usable', 'reject')),
    reason_tags jsonb NOT NULL,
    note text,
    preference_version_id uuid NOT NULL REFERENCES streamer_preference_versions(id),
    created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE streamer_preference_versions
    ADD CONSTRAINT streamer_preference_source_feedback_fk
    FOREIGN KEY (source_feedback_id) REFERENCES candidate_feedback(id);

ALTER TABLE candidate_jobs
    ADD COLUMN IF NOT EXISTS preference_version_id uuid
    REFERENCES streamer_preference_versions(id);

CREATE INDEX IF NOT EXISTS candidate_feedback_candidate_idx
    ON candidate_feedback (candidate_id, created_at);
