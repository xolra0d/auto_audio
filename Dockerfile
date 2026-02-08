FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

WORKDIR /app/src

ENV HF_TOKEN ${HF_TOKEN}
ENV BOT_TOKEN ${BOT_TOKEN}

CMD ["python3", "main.py"]
