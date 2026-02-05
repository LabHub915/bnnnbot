FROM python:3.9-slim

WORKDIR /app

# Install system dependencies if any needed (e.g. for ta-lib if used, but we use pandas)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .
# config.json will be mounted via docker-compose

CMD ["python", "-u", "bot.py"]
