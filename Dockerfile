# Multi-stage build for Flask + React Dashboard
# Stage 1: Build React Frontend
FROM node:18-alpine as frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm install --only=production

# Copy frontend source
COPY frontend/ ./

# Build React app
RUN npm run build

# Stage 2: Python Dependencies with Scientific Stack
FROM continuumio/miniconda3:latest as python-builder

# Set environment variables for faster builds
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTENSOR_FLAGS=device=cpu,floatX=float32,optimizer=fast_compile

# Set work directory
WORKDIR /app

# Install dependencies using conda (faster for scientific packages)
RUN conda update -n base -c defaults conda && \
    conda install -c conda-forge -y \
    python=3.9 \
    numpy \
    scipy \
    pandas \
    scikit-learn \
    flask \
    pyyaml \
    requests \
    python-dateutil && \
    conda clean -afy

# Install PyMC and remaining pip packages
RUN pip install --no-cache-dir \
    flask-cors==4.0.0 \
    pytensor>=2.10.0 \
    pymc>=5.0.0 \
    pydantic>=2.0.0 \
    loguru>=0.7.0 \
    yfinance>=0.2.0

# Stage 3: Production with Nginx + Flask (using conda base)
FROM continuumio/miniconda3:latest as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies for production
RUN apt-get update && apt-get install -y \
    nginx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Copy installed Python packages from conda builder stage
COPY --from=python-builder /opt/conda /opt/conda

# Copy built React app from frontend builder
COPY --from=frontend-builder /app/frontend/build /usr/share/nginx/html

# Copy application code
COPY src/ ./src/
COPY backend/ ./backend/
COPY data/ ./data/

# Copy built React app from frontend builder
COPY --from=frontend-builder /app/frontend/build /usr/share/nginx/html

# Copy nginx configuration
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Create necessary directories and set proper permissions
RUN mkdir -p /var/log/nginx /var/log/flask /var/lib/nginx/body /var/lib/nginx/fastcgi /var/lib/nginx/proxy /var/lib/nginx/scgi /var/lib/nginx/uwsgi /var/cache/nginx && \
    chown -R appuser:appuser /app /var/log/flask && \
    chown -R www-data:www-data /var/log/nginx /var/lib/nginx /var/cache/nginx /usr/share/nginx/html

# Expose ports
EXPOSE 80 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:80 || exit 1

# Start both nginx (as root) and flask (as appuser) services
CMD ["bash", "-c", "nginx && su appuser -c 'python backend/app.py'"]
