FROM python:3.11-slim

WORKDIR /app

# Install Python deps (pyproject includes sqlite-vec, which ships
# the vec0 loadable extension used for document vector search)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e "."

# Verify the sqlite-vec extension loads in this image so the
# vector index serves in "available" mode (not the cosine fallback)
RUN python -c "import sqlite3, sqlite_vec; db = sqlite3.connect(':memory:'); db.enable_load_extension(True); sqlite_vec.load(db); db.enable_load_extension(False); db.execute('CREATE VIRTUAL TABLE t USING vec0(id INTEGER PRIMARY KEY, e FLOAT[2] distance_metric=cosine)')"

# Copy application
COPY app/ ./app/
COPY main.py .
COPY docs/ ./docs/

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
