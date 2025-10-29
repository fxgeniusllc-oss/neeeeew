# Multi-stage Dockerfile for Dual Apex Core System
# Stage 1: Build Rust engine
FROM rust:1.75-slim as rust-builder

WORKDIR /build

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Rust source
COPY Cargo.toml .
COPY rust ./rust

# Build Rust library
RUN cargo build --release

# Stage 2: Python environment
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    postgresql-client \
    redis-tools \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Rust compiled library
COPY --from=rust-builder /build/target/release/*.so /usr/local/lib/python3.11/site-packages/

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Node.js dependencies
COPY package.json package-lock.json* ./
RUN npm install --production

# Copy application code
COPY python ./python
COPY node ./node
COPY frontend ./frontend
COPY config ./config
COPY .env.example .env

# Create necessary directories
RUN mkdir -p logs data models

# Expose ports
EXPOSE 8888 8889 8890

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8889/health || exit 1

# Start command (runs both Python orchestrator and Node API)
CMD ["sh", "-c", "node node/src/server.js & python3 python/orchestrator.py"]
