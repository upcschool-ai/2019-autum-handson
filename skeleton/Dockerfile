FROM python:3.5-slim

# Set the working directory to /app
WORKDIR /app

# Copy & install the requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy & install package
COPY MANIFEST.in /app
COPY setup.py /app
COPY requirements.txt /app
COPY skeleton /app/skeleton
RUN pip install .

# Run service
CMD skeleton-api
