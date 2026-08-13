FROM python:3.12-slim

WORKDIR /worker

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv export --only-group athena_worker --no-hashes --frozen -o requirements.txt \
    && uv pip install --system -r requirements.txt

COPY src/task_queue/services/workers/ /worker/task_queue/services/workers/
COPY src/task_queue/services/shared/ /worker/task_queue/services/shared/

ENV PYTHONPATH=/worker

CMD ["python", "task_queue/services/workers/athena/athena_worker.py"]