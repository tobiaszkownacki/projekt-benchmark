FROM python:3.12-slim

WORKDIR /poller

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

RUN uv export --only-group athena_poller --no-hashes --frozen -o requirements.txt \
    && uv pip install --system -r requirements.txt

COPY src/task_queue/services/pollers/ /poller/task_queue/services/pollers/
COPY src/shared/ /poller/shared/

RUN chmod +x /poller/task_queue/services/pollers/athena/init.sh

ENV PYTHONPATH=/poller

CMD ["/poller/task_queue/services/pollers/athena/init.sh"]