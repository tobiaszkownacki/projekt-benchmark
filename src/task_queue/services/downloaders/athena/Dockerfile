FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv export --only-group athena_downloader --no-hashes --frozen -o requirements.txt \
    && uv pip install --system -r requirements.txt

COPY src/task_queue/services/downloaders/ /app/task_queue/services/downloaders/
COPY src/shared/ /app/shared/

ENV PYTHONPATH=/app

CMD ["python", "task_queue/services/downloaders/athena/athena_downloader.py"]