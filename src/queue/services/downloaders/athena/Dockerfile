FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv export --only-group athena_downloader --no-hashes --frozen -o requirements.txt \
    && uv pip install --system -r requirements.txt

COPY src/queue/services/downloaders/ /app/queue/services/downloaders/
COPY src/queue/services/shared/ /app/queue/services/shared/

ENV PYTHONPATH=/app

CMD ["python", "queue/services/downloaders/athena/athena_downloader.py"]