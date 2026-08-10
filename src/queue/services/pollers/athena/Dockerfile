FROM python:3.12-slim

WORKDIR /poller

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

RUN uv export --only-group athena_poller --no-hashes --frozen -o requirements.txt \
    && uv pip install --system -r requirements.txt

COPY src/queue/services/pollers/ /poller/queue/services/pollers/
COPY src/queue/services/shared/ /poller/queue/services/shared/

RUN chmod +x /poller/queue/services/pollers/athena/init.sh

ENV PYTHONPATH=/poller

CMD ["/poller/queue/services/pollers/athena/init.sh"]