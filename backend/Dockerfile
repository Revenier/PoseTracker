# # Dockerfile

# FROM python:3.12-slim

# # Set working directory inside container
# WORKDIR /app

# # # Copy all files to container's /app folder
# COPY . .

# # # Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # # Jalankan file run.py
# # CMD ["python", "run.py"]

# # CMD service redis-server start && python run.py

# Gunakan base image Python
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    redis-server redis-tools && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY . .


EXPOSE 5000

CMD redis-server --daemonize yes --save "" --appendonly no && python run.py
