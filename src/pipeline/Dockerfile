FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv export --only-group athena-worker --no-hashes --frozen -o requirements.txt \
    && uv pip install --system -r requirements.txt

COPY src/shared/ /app/shared/
COPY src/pipeline/ /app/pipeline/

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-m", "pipeline.run"]
CMD ["worker"]