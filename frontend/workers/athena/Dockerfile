FROM python:3.12-slim

WORKDIR /worker

RUN pip install --no-cache-dir uv

COPY workers/athena/athena_worker_requirements.txt .
RUN uv pip install --system -r athena_worker_requirements.txt

COPY workers/__init__.py workers/__init__.py
COPY workers/abstract_worker.py workers/abstract_worker.py
COPY workers/logging_config.py workers/logging_config.py
COPY workers/athena/athena_worker.py workers/athena/athena_worker.py
COPY connectors/__init__.py connectors/__init__.py
COPY connectors/athena_connector.py connectors/athena_connector.py
COPY connectors/rabbitmq_connector.py connectors/rabbitmq_connector.py
COPY connectors/postgres_connector.py connectors/postgres_connector.py

ENV PYTHONPATH=/worker

CMD ["python", "workers/athena/athena_worker.py"]
