FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv export --only-group frontend --no-hashes --frozen -o requirements.txt \
    && uv pip install --system -r requirements.txt

COPY src/frontend/home.py ./
COPY src/frontend/auth/ ./auth/
COPY src/frontend/core/ ./core/
COPY src/frontend/views/ ./views/
COPY src/shared/ ./shared/

EXPOSE 8501

CMD ["streamlit", "run", "home.py", "--server.address=0.0.0.0", "--server.port=8501"]
