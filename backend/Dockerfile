# Use official Python image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
# Install PyTorch first (CPU version)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install the rest of dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Copy backend code
COPY fastapi_app/ ./fastapi_app
COPY model/ ./model
COPY *.py ./

# Expose FastAPI port
EXPOSE 8002

# Run FastAPI
CMD ["uvicorn", "fastapi_app.main:app", "--host", "0.0.0.0", "--port", "8002"]
