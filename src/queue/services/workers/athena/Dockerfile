FROM python:3.12-slim

WORKDIR /worker

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv export --only-group athena_worker --no-hashes --frozen -o requirements.txt \
    && uv pip install --system -r requirements.txt

COPY src/queue/services/workers/ /worker/queue/services/workers/
COPY src/queue/services/shared/ /worker/queue/services/shared/

ENV PYTHONPATH=/worker

CMD ["python", "queue/services/workers/athena/athena_worker.py"]