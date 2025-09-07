FROM python:3.11-slim

# System deps (add build tools if your model needs them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy API only first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source
COPY api /app/api
COPY models /app/models
COPY start.sh /app/start.sh

# If you *know* your models need heavy deps, you can add them here or rely on start.sh dynamic installs

EXPOSE 8080
CMD ["/app/start.sh"]
