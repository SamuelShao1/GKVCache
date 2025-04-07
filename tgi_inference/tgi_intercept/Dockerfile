FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install "uvicorn[standard]"
# Fastapi port
EXPOSE 9000

CMD ["uvicorn", "semantic_proxy:app", "--host", "0.0.0.0", "--port", "9000"]
