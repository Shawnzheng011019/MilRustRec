# Milvuso - Vector Database Recommendation System Demo

Milvuso is a lightweight recommendation system based on vector databases, developed in Rust with high performance, real-time updates, and scalable features.

## System Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│    User     │───▶│  Log Kafka   │───▶│ Feature Kafka   │
│   Actions   │    │              │    │                 │
└─────────────┘    └──────────────┘    └─────────────────┘
                           │                      │
                           ▼                      ▼
                   ┌──────────────┐    ┌─────────────────┐
                   │ Joiner Flink │───▶│Training Example │
                   │     Job      │    │     Kafka       │
                   └──────────────┘    └─────────────────┘
                                               │
                                               ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│ Model Server│◀───│ Training PS  │◀───│ Training Worker │
│             │    │              │    │                 │
└─────────────┘    └──────────────┘    └─────────────────┘
       │                   │                      │
       ▼                   ▼                      ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│ Serving PS  │    │Parameter Sync│    │Batch Training   │
│             │    │              │    │Data (HDFS)      │
└─────────────┘    └──────────────┘    └─────────────────┘
       │                   │
       ▼                   ▼
┌─────────────┐    ┌──────────────┐
│Vector DB    │    │    Redis     │
│ (Milvus)    │    │   Cache      │
└─────────────┘    └──────────────┘
```

## Core Features

- **High-Performance Algorithms**: Includes optimizers, retrievers, initializers and other high-performance components
- **Real-time User Profiling**: Build independent user profiles for each user with real-time recommendation updates
- **Vector Database**: Efficient vector similarity search based on Milvus
- **Stream Processing**: Real-time data stream processing using Kafka
- **Distributed Training**: Support for online learning and batch training
- **Multi-level Caching**: Redis cache + memory cache to improve response speed

## Technology Stack

- **Primary Language**: Rust
- **Vector Database**: Milvus
- **Message Queue**: Apache Kafka
- **Cache**: Redis
- **Database**: PostgreSQL
- **Web Framework**: Axum
- **Machine Learning**: Self-implemented collaborative filtering algorithms

## Quick Start

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f milvuso-server
```

### Local Development

1. **Install Dependencies**
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install pkg-config libssl-dev libsasl2-dev cmake build-essential
```

2. **Start Infrastructure**
```bash
# Start Milvus, Kafka, Redis, PostgreSQL
docker-compose up -d milvus kafka redis postgres
```

3. **Build and Run**
```bash
# Build project
cargo build --release

# Start recommendation server
./target/release/milvuso-server

# Start training worker
./target/release/milvuso-trainer

# Start feature generation worker
./target/release/milvuso-worker --worker-type feature
```

## API Usage Examples

### 1. Health Check
```bash
curl http://localhost:8080/health
```

### 2. Record User Actions
```bash
curl -X POST http://localhost:8080/actions \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "item_id": "550e8400-e29b-41d4-a716-446655440001",
    "action_type": "Click",
    "timestamp": "2024-01-01T12:00:00Z"
  }'
```

### 3. Get Recommendations
```bash
curl "http://localhost:8080/recommendations/550e8400-e29b-41d4-a716-446655440000?num_recommendations=10"
```

### 4. Add Item Features
```bash
curl -X POST http://localhost:8080/items \
  -H "Content-Type: application/json" \
  -d '{
    "item_id": "550e8400-e29b-41d4-a716-446655440001",
    "embedding": [0.1, 0.2, 0.3, ...],
    "category": "electronics",
    "tags": ["smartphone", "android"],
    "popularity_score": 0.8,
    "created_at": "2024-01-01T12:00:00Z"
  }'
```

## Configuration

The main configuration file is located at `config/default.toml`:

```toml
[server]
host = "0.0.0.0"
port = 8080

[milvus]
host = "localhost"
port = 19530
collection_name = "recommendation_vectors"
dimension = 128

[kafka]
brokers = "localhost:9092"
log_topic = "user_actions"
feature_topic = "features"
training_topic = "training_examples"

[recommendation]
embedding_dim = 128
top_k = 50
similarity_threshold = 0.7

[training]
batch_size = 1024
learning_rate = 0.001
negative_sampling_ratio = 4.0
```

## Core Algorithms

### 1. Collaborative Filtering Algorithm
- Uses matrix factorization techniques
- Supports SGD, Adam, AdaGrad and other optimizers
- Implements negative sampling and regularization

### 2. Vector Retrieval
- Supports cosine similarity and Euclidean distance
- Implements HNSW index for fast retrieval
- Dual retrieval support with memory and Milvus

### 3. Initialization Strategies
- Xavier, He, LeCun and other initialization methods
- Supports orthogonal initialization and sparse random initialization

## Performance Optimization

### 1. High-Performance Features
- Uses Rust zero-cost abstractions
- Parallel computing and SIMD optimization
- Memory pools and object reuse

### 2. Caching Strategy
- Three-level cache: Memory → Redis → Milvus
- LRU cache eviction policy
- Asynchronous preloading

### 3. Batch Processing Optimization
- Batch vector operations
- Pipeline processing
- Asynchronous I/O

## Monitoring and Metrics

The system provides rich monitoring metrics:

- **Recommendation Quality**: Precision@K, Recall@K, NDCG, MAP
- **Online Metrics**: CTR, conversion rate, engagement
- **System Metrics**: Latency, throughput, error rate
- **Business Metrics**: Coverage, diversity, novelty

## Scalability

### Horizontal Scaling
- Multiple recommendation service instances
- Kafka partition scaling
- Milvus cluster deployment

### Vertical Scaling
- GPU-accelerated training
- Larger vector dimensions
- More complex model architectures

## Testing

```bash
# Run unit tests
cargo test

# Run integration tests
cargo test --test integration

# Performance tests
cargo bench
```

## Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Create a Pull Request

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
