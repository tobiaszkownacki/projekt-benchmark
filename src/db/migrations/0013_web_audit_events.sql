-- Private operator audit for public page visits and authentication.
--
-- Keep this deliberately small: no request bodies, cookies or credentials.
-- The address is the client address reconstructed from the trusted reverse
-- proxy headers, and the user agent is bounded for storage safety.
CREATE TABLE IF NOT EXISTS web_audit_events (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_type   TEXT NOT NULL CHECK (event_type IN ('page_view', 'login_success', 'login_failure')),
    occurred_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id      UUID REFERENCES users (id) ON DELETE SET NULL,
    client_ip    INET,
    path         TEXT NOT NULL,
    method       TEXT NOT NULL,
    status_code  SMALLINT,
    user_agent   TEXT
);

CREATE INDEX IF NOT EXISTS idx_web_audit_events_occurred
    ON web_audit_events (occurred_at DESC);

CREATE INDEX IF NOT EXISTS idx_web_audit_events_type_occurred
    ON web_audit_events (event_type, occurred_at DESC);

CREATE INDEX IF NOT EXISTS idx_web_audit_events_user_occurred
    ON web_audit_events (user_id, occurred_at DESC)
    WHERE user_id IS NOT NULL;
