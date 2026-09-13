# Two stages: Node builds the frontend, and does not appear in the result.
#
# A build-time bundler avoids needing a Node process at runtime, which would
# add a second runtime to patch for no benefit here.
FROM node:22-alpine AS frontend

WORKDIR /build
COPY src/frontend/package.json src/frontend/package-lock.json ./
RUN npm ci
COPY src/frontend/ ./

# Empty by default, so the image serves the application at the origin root.
# Set to a prefix (e.g. "/benchmark/") to deploy behind a path on a shared host;
# it has to be baked in here because the bundler rewrites asset URLs.
ARG BASE_PATH=
ENV BASE_PATH=$BASE_PATH
RUN npm run build


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install --no-cache-dir \
        "fastapi>=0.115" "uvicorn[standard]>=0.32" "psycopg[binary,pool]>=3.2" \
        "itsdangerous>=2.2" "python-multipart>=0.0.12" "bcrypt>=4.1" \
        "httpx>=0.27" "authlib>=1.3" "pika>=1.3" "email-validator>=2.1"

COPY src/backend       /app/backend
COPY src/db/migrations /app/migrations
# Read at runtime by /docs, so the protocol page shows the live contract rather
# than a copy that drifts.
COPY src/benchmark_core /app/src/benchmark_core

COPY --from=frontend /build/dist /app/static

ENV STATIC_ROOT=/app/static \
    MIGRATIONS_DIR=/app/migrations \
    ARTIFACT_ROOT=/downloads \
    PYTHONPATH=/app

RUN useradd --system --uid 10001 --create-home benchmark
USER benchmark

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=4s --start-period=25s --retries=4 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).status==200 else 1)"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
