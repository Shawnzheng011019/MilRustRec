#!/bin/bash

# MilRustRec Recommendation System Startup Script

set -e

echo "🚀 Starting MilRustRec Recommendation System"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed, please install Docker first"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed, please install Docker Compose first"
    exit 1
fi

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "❌ Rust is not installed, please install Rust first"
    echo "Run: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

echo "✅ Environment check passed"

# Create necessary directories
mkdir -p data/milvus
mkdir -p data/kafka
mkdir -p data/redis
mkdir -p data/postgres
mkdir -p logs

echo "📁 Created data directories"

# Start infrastructure services
echo "🔧 Starting infrastructure services..."
docker-compose up -d etcd minio milvus zookeeper kafka redis postgres

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 30

# Check service status
echo "🔍 Checking service status..."

# Check Kafka
if docker exec kafka kafka-topics --list --bootstrap-server localhost:9092 &> /dev/null; then
    echo "✅ Kafka service is running"
else
    echo "❌ Kafka service is not running"
    exit 1
fi

# Check Redis
if docker exec redis redis-cli ping | grep -q PONG; then
    echo "✅ Redis service is running"
else
    echo "❌ Redis service is not running"
    exit 1
fi

# Check PostgreSQL
if docker exec postgres pg_isready -U MilRustRec &> /dev/null; then
    echo "✅ PostgreSQL service is running"
else
    echo "❌ PostgreSQL service is not running"
    exit 1
fi

# Create Kafka topics
echo "📨 Creating Kafka topics..."
docker exec kafka kafka-topics --create --topic user_actions --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1 --if-not-exists
docker exec kafka kafka-topics --create --topic features --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1 --if-not-exists
docker exec kafka kafka-topics --create --topic training_examples --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1 --if-not-exists

echo "✅ Kafka topics created successfully"

# Build Rust project
echo "🔨 Building Rust project..."
cargo build --release

echo "✅ Build completed"

# Start recommendation server
echo "🚀 Starting recommendation server..."
RUST_LOG=info ./target/release/milvuso-server &
SERVER_PID=$!

# Wait for server to start
sleep 5

# Health check
if curl -f http://localhost:8080/health &> /dev/null; then
    echo "✅ Recommendation server started successfully"
else
    echo "❌ Recommendation server failed to start"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

echo "🎉 MilRustRec Recommendation System started successfully!"
echo ""
echo "📊 Service Status:"
echo "  - Recommendation Server: http://localhost:8080"
echo "  - Health Check: http://localhost:8080/health"
echo "  - Milvus: localhost:19530"
echo "  - Kafka: localhost:9092"
echo "  - Redis: localhost:6379"
echo "  - PostgreSQL: localhost:5432"
echo ""
echo "📖 Usage Examples:"
echo "  # Health check"
echo "  curl http://localhost:8080/health"
echo ""
echo "  # Record user action"
echo "  curl -X POST http://localhost:8080/actions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"user_id\":\"550e8400-e29b-41d4-a716-446655440000\",\"item_id\":\"550e8400-e29b-41d4-a716-446655440001\",\"action_type\":\"Click\",\"timestamp\":\"2024-01-01T12:00:00Z\"}'"
echo ""
echo "  # Get recommendations"
echo "  curl 'http://localhost:8080/recommendations/550e8400-e29b-41d4-a716-446655440000?num_recommendations=10'"
echo ""
echo "🛑 Stop services:"
echo "  kill $SERVER_PID"
echo "  docker-compose down"

# Save PID to file
echo $SERVER_PID > milvuso.pid

echo ""
echo "💡 Tip: Server PID saved to milvuso.pid file"
echo "Use 'kill \$(cat milvuso.pid)' to stop the server"
