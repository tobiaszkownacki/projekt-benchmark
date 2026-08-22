FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv export --only-group api-fastapi --no-hashes --frozen -o requirements.txt \
    && uv pip install --system -r requirements.txt

COPY src/task_queue/services/api/ /app/task_queue/services/api/
COPY src/shared/ /app/shared/

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "task_queue.services.api.fastapi.main:app", "--host", "0.0.0.0", "--port", "8000"]
