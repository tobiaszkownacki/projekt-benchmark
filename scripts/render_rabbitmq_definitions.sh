#!/usr/bin/env bash
# Render src/task_queue/config/rabbitmq/definitions.json from its template.
#
# docker-compose.yml mounts definitions.json, but that file is gitignored
# (correctly -- it carries the broker password) and only the template is in the
# repository. On a fresh clone `docker compose up` therefore has nothing to
# mount, and Docker helpfully creates a *directory* with that name, after which
# RabbitMQ fails to start with an error that does not mention any of this.
#
# Run this once after writing .env.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
template="$root/src/task_queue/config/rabbitmq/definitions.template.json"
output="$root/src/task_queue/config/rabbitmq/definitions.json"
env_file="$root/.env"

if [[ ! -f "$env_file" ]]; then
    echo "error: $env_file not found. Copy .env.template and fill it in first." >&2
    exit 1
fi

set -a
# shellcheck disable=SC1090
. "$env_file"
set +a

: "${RABBITMQ_USER:?RABBITMQ_USER is not set in .env}"
: "${RABBITMQ_PASSWORD:?RABBITMQ_PASSWORD is not set in .env}"

python3 - "$template" "$output" <<'PY'
import json
import os
import sys

template_path, output_path = sys.argv[1], sys.argv[2]
raw = open(template_path, encoding="utf-8").read()
raw = raw.replace("[USER]", os.environ["RABBITMQ_USER"])
raw = raw.replace("[PASSWORD]", os.environ["RABBITMQ_PASSWORD"])

json.loads(raw)  # fail loudly here rather than inside the broker at boot
open(output_path, "w", encoding="utf-8").write(raw)
PY

chmod 600 "$output"
echo "wrote $output"
