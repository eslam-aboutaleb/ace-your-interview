FROM python:3.11-slim

WORKDIR /app

# Install Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir -e "."

# Copy application
COPY app/ ./app/
COPY main.py .
COPY docs/ ./docs/

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
