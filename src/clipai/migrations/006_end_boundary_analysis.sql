ALTER TABLE clip_candidates
    ADD COLUMN IF NOT EXISTS boundary_analysis jsonb NOT NULL DEFAULT '{}'::jsonb;
