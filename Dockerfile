# Use official Python base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy all local files to the container
COPY . .
COPY data /app/data

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
ENV PYTHONPATH=/app

# Run prediction script
CMD ["python", "src/predict.py"]
