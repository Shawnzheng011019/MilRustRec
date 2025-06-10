# Build stage
FROM rust:1.75-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libsasl2-dev \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src ./src

# Build the application
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libsasl2-2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the binary from builder stage
COPY --from=builder /app/target/release/milvuso-server ./
COPY --from=builder /app/target/release/milvuso-trainer ./
COPY --from=builder /app/target/release/milvuso-worker ./

# Copy configuration
COPY config ./config

# Create a non-root user
RUN useradd -r -s /bin/false milvuso
RUN chown -R milvuso:MilRustRec /app
USER milvuso

# Expose port
EXPOSE 8080

# Default command
CMD ["./milvuso-server"]
