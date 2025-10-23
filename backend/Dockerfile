# Dockerfile

FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# # Copy all files to container's /app folder
COPY . .

# # Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# # Jalankan file run.py
CMD ["python", "run.py"]

# WORKDIR /app

# COPY backend/requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY backend/ .

# CMD ["python", "run.py"]
