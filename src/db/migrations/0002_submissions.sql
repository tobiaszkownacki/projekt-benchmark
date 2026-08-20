-- Optimizer submissions: the code a participant sends, and what the validator
-- said about it. BRIEF.md §10 lists this as missing from the data model.

DO $$ BEGIN
    CREATE TYPE optimizer_family_t AS ENUM ('gradient', 'gradient_free');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE submission_status_t AS ENUM ('validating', 'rejected', 'accepted');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE TABLE IF NOT EXISTS submissions (
    submission_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submitted_by      UUID NOT NULL REFERENCES users (id),
    display_name      TEXT NOT NULL,
    kind              TEXT NOT NULL CHECK (kind IN ('builtin', 'uploaded')),
    builtin_name      TEXT,
    source_code       TEXT,
    source_sha256     CHAR(64),
    output_type       TEXT,
    family            optimizer_family_t,
    status            submission_status_t NOT NULL DEFAULT 'validating',
    validator_log     TEXT,
    validator_version TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    validated_at      TIMESTAMPTZ,

    CONSTRAINT builtin_xor_source CHECK (
        (kind = 'builtin'  AND builtin_name IS NOT NULL AND source_code IS NULL) OR
        (kind = 'uploaded' AND source_code  IS NOT NULL AND source_sha256 IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_submissions_user
    ON submissions (submitted_by, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_submissions_sha
    ON submissions (source_sha256) WHERE source_sha256 IS NOT NULL;
