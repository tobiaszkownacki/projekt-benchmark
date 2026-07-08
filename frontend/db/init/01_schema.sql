CREATE TYPE user_role AS ENUM ('unverified', 'verified', 'admin');

CREATE TYPE auth_provider AS ENUM ('email', 'google', 'microsoft');

CREATE TABLE users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           VARCHAR(255) NOT NULL UNIQUE,
    password_hash   VARCHAR(255),
    role            user_role NOT NULL DEFAULT 'unverified',
    auth_provider   auth_provider NOT NULL,
    oauth_sub       VARCHAR(255),
    display_name    VARCHAR(255),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login_at   TIMESTAMPTZ,

    CONSTRAINT uq_oauth_identity UNIQUE (auth_provider, oauth_sub),
    CONSTRAINT chk_email_password CHECK (
        auth_provider != 'email' OR password_hash IS NOT NULL
    )
);

CREATE INDEX idx_users_email ON users (email);
CREATE INDEX idx_users_role ON users (role);
