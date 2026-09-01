#!/usr/bin/env bash
#
# Refuses a commit that would place a secret in the repository.
#
# .gitignore already covers the files we know about, and nothing sensitive has
# ever been committed here. This guards the case .gitignore cannot: a value
# pasted into a file that is meant to be tracked -- a README, a compose
# override, a test fixture. That is how secrets usually leak, not through the
# file everyone remembers to ignore.
#
# Install as a hook:  ./scripts/check-secrets.sh --install
# Run by hand:        ./scripts/check-secrets.sh

set -uo pipefail
cd "$(dirname "$0")/.."

if [ "${1:-}" = "--install" ]; then
    hook=$(git rev-parse --git-path hooks/pre-commit)
    printf '#!/bin/sh\nexec ./scripts/check-secrets.sh\n' > "$hook"
    chmod +x "$hook"
    echo "installed $hook"
    exit 0
fi

# Files about to be committed, minus deletions.
staged=$(git diff --cached --name-only --diff-filter=ACM)
[ -z "$staged" ] && exit 0

fail=0
note() { echo "  $*" >&2; }

# 1. A tracked file that should never be tracked at all.
while IFS= read -r f; do
    case "$f" in
        .env|.env.*|*.seed-credentials|.seed-credentials|*secrets.toml|*.pem|*.key|id_rsa*)
            [ "$f" = ".env.template" ] && continue
            [ $fail -eq 0 ] && echo "blocked: secret material staged" >&2
            note "$f -- this file must never be committed"
            fail=1
            ;;
    esac
done <<< "$staged"

# 2. A secret-shaped assignment inside a file that IS meant to be tracked.
#    Matches an assignment whose right-hand side is a non-empty literal; an
#    empty value (.env.template) and a shell interpolation (${VAR}, $(cmd))
#    are both fine, which is what makes this usable as a blocking hook.
pattern='(PASSWORD|SECRET|TOKEN|API_KEY|CLIENT_SECRET|PRIVATE_KEY)[A-Z_]*[[:space:]]*[=:][[:space:]]*["'"'"']?[A-Za-z0-9/+_.-]{8,}'
while IFS= read -r f; do
    [ -f "$f" ] || continue
    case "$f" in
        scripts/check-secrets.sh|*.lock|*.md) continue ;;
    esac
    hits=$(git show ":$f" 2>/dev/null \
        | grep -nEI "$pattern" \
        | grep -vE '\$\{|\$\(|<set>|example|changeme|xxx|\.\.\.' || true)
    if [ -n "$hits" ]; then
        [ $fail -eq 0 ] && echo "blocked: secret-shaped value in a tracked file" >&2
        note "$f:"
        echo "$hits" | sed 's/^/    /' >&2
        fail=1
    fi
done <<< "$staged"

if [ $fail -ne 0 ]; then
    echo >&2
    echo "Nothing was committed. Move the value into .env (gitignored) and read it" >&2
    echo "from the environment, or run scripts/bootstrap-env.sh to generate one." >&2
    exit 1
fi
exit 0
