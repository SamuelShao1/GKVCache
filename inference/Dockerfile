FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Persistent mdels dir
RUN mkdir -p /models

EXPOSE 8001

# Start the inference worker
CMD ["uvicorn", "worker:app", "--host", "0.0.0.0", "--port", "8001"]