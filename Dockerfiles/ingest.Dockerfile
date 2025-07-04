# For GPU support (default).
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
# For CPU-only, comment the line above and uncomment the line below.
# FROM python:3.11-slim
WORKDIR /app

COPY requirements/ /app/requirements/

RUN pip install --no-cache-dir -r requirements/ingest.txt

COPY ingest.py ./

COPY gcloud_credentials.json /app/gcloud_credentials.json
ENV GOOGLE_APPLICATION_CREDENTIALS /app/gcloud_credentials.json

CMD ["sh", "-c", "while true; do python ingest.py; sleep 3600; done"]
