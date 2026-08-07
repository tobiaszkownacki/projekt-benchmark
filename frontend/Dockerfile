FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-install-project

COPY . .

EXPOSE 8501

CMD ["uv", "run", "--no-sync", "streamlit", "run", "home.py", "--server.address=0.0.0.0", "--server.port=8501"]
