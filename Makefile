# MilRustRec Recommendation System Makefile

.PHONY: help build test clean run docker-build docker-up docker-down fmt clippy bench docs

# Default target
help:
	@echo "MilRustRec Recommendation System Build Tool"
	@echo ""
	@echo "Available commands:"
	@echo "  build        - Build project"
	@echo "  test         - Run tests"
	@echo "  clean        - Clean build files"
	@echo "  run          - Run recommendation server"
	@echo "  run-trainer  - Run training worker"
	@echo "  run-worker   - Run feature worker"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-up    - Start all services"
	@echo "  docker-down  - Stop all services"
	@echo "  fmt          - Format code"
	@echo "  clippy       - Run Clippy checks"
	@echo "  bench        - Run performance tests"
	@echo "  docs         - Generate documentation"
	@echo "  example      - Run basic example"

# Build project
build:
	@echo "🔨 Building project..."
	cargo build --release

# Run tests
test:
	@echo "🧪 Running tests..."
	cargo test

# Clean build files
clean:
	@echo "🧹 Cleaning build files..."
	cargo clean

# Run recommendation server
run:
	@echo "🚀 Starting recommendation server..."
	cargo run --bin milvuso-server

# Run training worker
run-trainer:
	@echo "🎓 Starting training worker..."
	cargo run --bin milvuso-trainer

# Run feature worker
run-worker:
	@echo "⚙️ Starting feature worker..."
	cargo run --bin milvuso-worker -- --worker-type feature

# Run joiner worker
run-joiner:
	@echo "🔗 Starting joiner worker..."
	cargo run --bin milvuso-worker -- --worker-type joiner

# Build Docker image
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t milvuso:latest .

# Start all services
docker-up:
	@echo "🚀 Starting all services..."
	docker-compose up -d

# Stop all services
docker-down:
	@echo "🛑 Stopping all services..."
	docker-compose down

# Check service status
docker-status:
	@echo "📊 Checking service status..."
	docker-compose ps

# View logs
docker-logs:
	@echo "📋 Viewing service logs..."
	docker-compose logs -f

# Format code
fmt:
	@echo "✨ Formatting code..."
	cargo fmt

# Run Clippy checks
clippy:
	@echo "📎 Running Clippy checks..."
	cargo clippy -- -D warnings

# Run performance tests
bench:
	@echo "⚡ Running performance tests..."
	cargo bench

# Generate documentation
docs:
	@echo "📚 Generating documentation..."
	cargo doc --open

# Run basic example
example:
	@echo "💡 Running basic example..."
	cargo run --example basic_usage

# Check code quality
check: fmt clippy test
	@echo "✅ Code quality check completed"

# Complete build process
all: clean build test
	@echo "🎉 Complete build process finished"

# Development environment setup
dev-setup:
	@echo "🛠️ Setting up development environment..."
	@echo "Installing Rust toolchain..."
	rustup component add rustfmt clippy
	@echo "Installing cargo-watch..."
	cargo install cargo-watch
	@echo "✅ Development environment setup completed"

# Watch file changes and auto-recompile
watch:
	@echo "👀 Watching file changes..."
	cargo watch -x "build --release"

# Watch file changes and auto-run tests
watch-test:
	@echo "👀 Watching file changes and running tests..."
	cargo watch -x test

# Performance profiling
profile:
	@echo "📊 Running performance profiling..."
	cargo build --release
	perf record --call-graph=dwarf ./target/release/milvuso-server
	perf report

# Memory check
valgrind:
	@echo "🔍 Running memory check..."
	cargo build
	valgrind --tool=memcheck --leak-check=full ./target/debug/milvuso-server

# Security audit
audit:
	@echo "🔒 Running security audit..."
	cargo audit

# Update dependencies
update:
	@echo "📦 Updating dependencies..."
	cargo update

# Generate coverage report
coverage:
	@echo "📈 Generating coverage report..."
	cargo tarpaulin --out Html

# Database migration
migrate:
	@echo "🗄️ Running database migration..."
	sqlx migrate run

# Create Kafka topics
kafka-topics:
	@echo "📨 Creating Kafka topics..."
	docker exec kafka kafka-topics --create --topic user_actions --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
	docker exec kafka kafka-topics --create --topic features --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
	docker exec kafka kafka-topics --create --topic training_examples --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

# Health check
health-check:
	@echo "🏥 Running health check..."
	curl -f http://localhost:8080/health || echo "Server not running"

# Load test
load-test:
	@echo "🚛 Running load test..."
	@echo "Need to install wrk: sudo apt-get install wrk"
	wrk -t12 -c400 -d30s http://localhost:8080/health

# Clean Docker resources
docker-clean:
	@echo "🧹 Cleaning Docker resources..."
	docker-compose down -v
	docker system prune -f

# Backup data
backup:
	@echo "💾 Backing up data..."
	docker exec postgres pg_dump -U MilRustRec MilRustRec > backup_$(shell date +%Y%m%d_%H%M%S).sql

# Restore data
restore:
	@echo "🔄 Restoring data..."
	@read -p "Enter backup filename: " backup_file; \
	docker exec -i postgres psql -U MilRustRec MilRustRec < $$backup_file

# Generate API documentation
api-docs:
	@echo "📖 Generating API documentation..."
	@echo "Visit http://localhost:8080/docs after starting the server"

# Deploy to production
deploy:
	@echo "🚀 Deploying to production..."
	@echo "Please ensure production configuration files are properly set up"
	docker-compose -f docker-compose.prod.yml up -d

# Create release
release:
	@echo "🏷️ Creating release..."
	@read -p "Enter version number (e.g. v1.0.0): " version; \
	git tag $$version && git push origin $$version
