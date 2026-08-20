-- Bearer tokens for programmatic submission.
--
-- §15 makes this a product requirement rather than tidiness: a GECCO entrant
-- submits from a CLI or from CI, not from a browser form. Only the hash is
-- stored, so a database dump does not yield usable tokens.

CREATE TABLE IF NOT EXISTS api_tokens (
    token_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    name         TEXT NOT NULL,
    token_sha256 CHAR(64) NOT NULL UNIQUE,
    prefix       TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at TIMESTAMPTZ,
    revoked_at   TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_tokens_user ON api_tokens (user_id);
