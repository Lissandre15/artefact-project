# Python 3.10.6 as base image
FROM python:3.10.6

# Set working directory
WORKDIR /flask-app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
COPY requirements.txt .

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy application files
COPY . /flask-app

# Expose port 8080
EXPOSE 8080

# Set environment variables
ENV PYTHONPATH=/flask-app
ENV FLASK_APP=/flask-app/app.py

# Start Flask
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]