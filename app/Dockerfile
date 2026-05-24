# =============================================================
# Dockerfile — Heart Disease Prediction API
# =============================================================
# This file tells Docker how to build a container for our API.
# Think of it as a recipe: start with Python, add our code,
# install libraries, then run the API server.

# Start from official Python 3.10 image (lightweight version)
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements file first (speeds up rebuilds)
COPY requirements.txt .

# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files needed for predictions
COPY heart_disease_model.pkl .
COPY feature_columns.pkl .

# Copy the FastAPI application
COPY app/main.py .

# Expose port 8000 (where the API will listen)
EXPOSE 8000

# Command to start the API server when container runs
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
