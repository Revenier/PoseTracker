# Dockerfile

FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# # Copy all files to container's /app folder
COPY . .

# # Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# # Jalankan file run.py
CMD ["python", "run.py"]

