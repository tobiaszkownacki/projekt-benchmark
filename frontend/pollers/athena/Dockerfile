FROM python:3.12-slim

WORKDIR /poller

RUN pip install --no-cache-dir uv

COPY pollers/athena/athena_poller_requirements.txt /poller/.

RUN uv pip install --system -r athena_poller_requirements.txt

COPY pollers/athena/init.sh /poller/.
COPY pollers/logging_config.py /poller/.
COPY pollers/abstract_poller.py /poller/.
COPY connectors/ /poller/.
COPY pollers/athena/athena_poller.py /poller/.