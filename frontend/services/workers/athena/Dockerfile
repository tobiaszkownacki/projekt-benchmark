FROM python:3.12-slim

WORKDIR /worker

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv export --only-group athena_worker --no-hashes --frozen -o requirements.txt \
    && uv pip install --system -r requirements.txt

COPY services/workers/__init__.py workers/__init__.py
COPY services/workers/abstract_worker.py workers/abstract_worker.py
COPY services/workers/logging_config.py workers/logging_config.py
COPY services/workers/athena/athena_worker.py workers/athena/athena_worker.py
COPY connectors/__init__.py connectors/__init__.py
COPY connectors/athena_connector.py connectors/athena_connector.py
COPY connectors/rabbitmq_connector.py connectors/rabbitmq_connector.py
COPY connectors/postgres_connector.py connectors/postgres_connector.py

ENV PYTHONPATH=/worker

CMD ["python", "workers/athena/athena_worker.py"]
