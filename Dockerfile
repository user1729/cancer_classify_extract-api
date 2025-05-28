# Use official CUDA 12.1 image with Ubuntu
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy Python dependencies
ADD requirements.txt .
RUN pip3 install -r requirements.txt 

# Expose port
EXPOSE 8000

# Copy application code
COPY app ./app
COPY models/fine_tuned ./models/fine_tuned

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
