FROM python:3.5-slim

# Set the working directory to /app
WORKDIR /app

# Install system deps
RUN apt-get update && \
    apt-get install -y libgtk2.0-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy & install the requirements
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Copy & install package
COPY MANIFEST.in /app
COPY setup.py /app
COPY requirements.txt /app
COPY skeleton /app/skeleton
RUN pip install .

# Run service
CMD skeleton-api
