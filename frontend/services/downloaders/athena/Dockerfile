FROM python:3.12-slim

WORKDIR /downloader

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv export --only-group athena_downloader --no-hashes --frozen -o requirements.txt \
    && uv pip install --system -r requirements.txt

COPY services/downloaders/__init__.py downloaders/__init__.py
COPY services/downloaders/abstract_downloader.py downloaders/abstract_downloader.py
COPY services/downloaders/logging_config.py downloaders/logging_config.py
COPY services/downloaders/athena/athena_downloader.py downloaders/athena/athena_downloader.py
COPY connectors/__init__.py connectors/__init__.py
COPY connectors/athena_connector.py connectors/athena_connector.py
COPY connectors/rabbitmq_connector.py connectors/rabbitmq_connector.py
COPY connectors/postgres_connector.py connectors/postgres_connector.py

ENV PYTHONPATH=/downloader

CMD ["python", "downloaders/athena/athena_downloader.py"]
