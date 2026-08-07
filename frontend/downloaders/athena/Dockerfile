FROM python:3.12-slim

WORKDIR /downloader

RUN pip install --no-cache-dir uv

COPY downloaders/athena/athena_downloader_requirements.txt .
RUN uv pip install --system -r athena_downloader_requirements.txt

COPY downloaders/__init__.py downloaders/__init__.py
COPY downloaders/abstract_downloader.py downloaders/abstract_downloader.py
COPY downloaders/logging_config.py downloaders/logging_config.py
COPY downloaders/athena/athena_downloader.py downloaders/athena/athena_downloader.py
COPY connectors/__init__.py connectors/__init__.py
COPY connectors/athena_connector.py connectors/athena_connector.py
COPY connectors/rabbitmq_connector.py connectors/rabbitmq_connector.py
COPY connectors/postgres_connector.py connectors/postgres_connector.py

ENV PYTHONPATH=/downloader

CMD ["python", "downloaders/athena/athena_downloader.py"]
