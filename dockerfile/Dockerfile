# Dockerfile for FastAPI churn prediction API
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model
COPY src/ src/
COPY models/ models/

# Expose port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "src.serve_api:app", "--host", "0.0.0.0", "--port", "8000"]
