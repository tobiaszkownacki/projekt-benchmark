FROM python:3.12-slim

WORKDIR /poller

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock /poller/.

RUN uv export --only-group athena_poller --no-hashes --frozen -o requirements.txt \
    && uv pip install --system -r requirements.txt

COPY services/pollers/athena/init.sh /poller/.
COPY services/pollers/logging_config.py /poller/.
COPY services/pollers/abstract_poller.py /poller/.
COPY connectors/ /poller/.
COPY services/pollers/athena/athena_poller.py /poller/.