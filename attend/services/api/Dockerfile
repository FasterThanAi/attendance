FROM python:3.11-slim

WORKDIR /app

# ffmpeg is not needed in the api container (only the worker touches video),
# so keep this image light.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
