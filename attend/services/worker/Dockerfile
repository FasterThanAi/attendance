FROM python:3.11-slim

WORKDIR /app

# ffmpeg is required from Phase 3 onward for frame extraction. Installed now
# so later phases don't need a Dockerfile change.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["rq", "worker", "--url", "redis://redis:6379/0", "attend"]
