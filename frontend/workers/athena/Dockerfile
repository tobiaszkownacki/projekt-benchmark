FROM python:3.11-slim

WORKDIR /worker

RUN pip install --no-cache-dir uv

COPY workers/athena/athena_requirements.txt .

RUN uv pip install --system -r athena_requirements.txt

COPY workers/athena/athena_worker.py .

CMD ["python","athena_worker.py"]
