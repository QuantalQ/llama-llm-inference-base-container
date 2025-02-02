FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    git cmake build-essential wget curl \
 && rm -rf /var/lib/apt/lists/*

#  Create a directory for your model weights (we will download them at runtime or build time)
RUN mkdir /app/model
WORKDIR /app

#  Copy your inference script, which starts a web server (Flask/FastAPI).
COPY app.py /app/app.py
# Copy the requirements file first
COPY requirements.txt /app
# Upgrade pip and install Python dependencies from requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

#  Expose port 8080 (SageMaker usually forwards to 8080 for inference).
EXPOSE 8080

#  Set the entrypoint to run a simple server
ENTRYPOINT ["python", "/app/app.py"]
