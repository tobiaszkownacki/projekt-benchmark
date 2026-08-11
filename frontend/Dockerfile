FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy project specification files
COPY pyproject.toml uv.lock ./
# Install third-party dependencies first for Docker caching
RUN uv sync --frozen --no-cache --no-install-project

# Copy application source code
COPY . .

# Install root project package
RUN uv sync --frozen --no-cache

EXPOSE 8501

HEALTHCHECK CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["uv", "run", "streamlit", "run", "home.py", "--server.port=8501", "--server.address=0.0.0.0"]
