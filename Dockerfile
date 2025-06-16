# FROM python:3.12-slim

# WORKDIR /app

# # Install system dependencies for langchain-astradb and other packages
# RUN apt-get update && apt-get install -y \
#     gcc \
#     libc-dev \
#     && rm -rf /var/lib/apt/lists/*

# COPY ./requirements.txt /app/requirements.txt
# RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# COPY . /app

# EXPOSE 8000

# CMD ["fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]



# FROM python:3.12-slim AS builder

# # Create a non-privileged user
# RUN addgroup --system appgroup && adduser --system --group appuser

# # Install build dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     gcc \
#     libc-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Set up virtualenv
# RUN python -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# # Install Python dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir --upgrade -r requirements.txt

# # Final stage
# FROM python:3.12-slim

# # Create a non-privileged user
# RUN addgroup --system appgroup && adduser --system --group appuser

# # Copy virtual environment from builder stage
# COPY --from=builder /opt/venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# # Set working directory and copy application code
# WORKDIR /app
# COPY . /app

# # Set environment variables
# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     PORT=8000

# # Switch to non-root user
# USER appuser

# # Expose port
# EXPOSE 8000

# # Health check
# HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1

# # Command to run the application
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM python:3.12-slim AS builder

# Create a non-privileged user
RUN addgroup --system appgroup && adduser --system --group appuser

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# Final stage
FROM python:3.12-slim

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-privileged user
RUN addgroup --system appgroup && adduser --system --group appuser

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory and copy application code
WORKDIR /app
COPY . /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application with Gunicorn
# Workers = (2 * CPU cores) + 1
CMD ["gunicorn", "new:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
