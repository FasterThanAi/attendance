FROM python:3.11-slim

WORKDIR /app

# Phase 2 update: ffprobe (part of ffmpeg) is needed here after all, to
# validate an assembled upload's duration/resolution/codec before creating a
# video_upload row -- that's plain metadata inspection, not ML work, so it's
# a fine fit for the otherwise-light api image (unlike the actual ML stack,
# which stays worker-only).
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
