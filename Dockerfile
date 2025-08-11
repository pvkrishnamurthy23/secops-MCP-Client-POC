# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p /app/logs

# Expose Streamlit port
EXPOSE 80

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=80
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:80/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=80", "--server.address=0.0.0.0"] 