FROM python:3.11-slim

WORKDIR /app

COPY requirements/ /app/requirements/

RUN pip install --no-cache-dir -r requirements/bot.txt

COPY bot.py graph.py ./

COPY gcloud_credentials.json /app/gcloud_credentials.json
ENV GOOGLE_APPLICATION_CREDENTIALS /app/gcloud_credentials.json

CMD ["python", "bot.py"]