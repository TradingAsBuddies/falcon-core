FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip

COPY . .
RUN pip install --no-cache-dir --timeout 120 ".[full]"
