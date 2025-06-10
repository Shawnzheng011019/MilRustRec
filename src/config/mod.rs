use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub milvus: MilvusConfig,
    pub kafka: KafkaConfig,
    pub redis: RedisConfig,
    pub postgres: PostgresConfig,
    pub recommendation: RecommendationConfig,
    pub training: TrainingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub workers: usize,
}

impl ServerConfig {
    pub fn socket_addr(&self) -> SocketAddr {
        format!("{}:{}", self.host, self.port).parse().unwrap()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilvusConfig {
    pub host: String,
    pub port: u16,
    pub collection_name: String,
    pub dimension: usize,
    pub index_type: String,
    pub metric_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KafkaConfig {
    pub brokers: String,
    pub log_topic: String,
    pub feature_topic: String,
    pub training_topic: String,
    pub group_id: String,
    pub auto_offset_reset: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    pub url: String,
    pub pool_size: u32,
    pub ttl_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostgresConfig {
    pub url: String,
    pub max_connections: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationConfig {
    pub embedding_dim: usize,
    pub top_k: usize,
    pub similarity_threshold: f32,
    pub user_profile_update_interval: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub learning_rate: f64,
    pub epochs: usize,
    pub model_save_interval: u64,
    pub negative_sampling_ratio: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 8080,
                workers: num_cpus::get(),
            },
            milvus: MilvusConfig {
                host: "localhost".to_string(),
                port: 19530,
                collection_name: "recommendation_vectors".to_string(),
                dimension: 128,
                index_type: "IVF_FLAT".to_string(),
                metric_type: "L2".to_string(),
            },
            kafka: KafkaConfig {
                brokers: "localhost:9092".to_string(),
                log_topic: "user_actions".to_string(),
                feature_topic: "features".to_string(),
                training_topic: "training_examples".to_string(),
                group_id: "milvuso_group".to_string(),
                auto_offset_reset: "earliest".to_string(),
            },
            redis: RedisConfig {
                url: "redis://localhost:6379".to_string(),
                pool_size: 10,
                ttl_seconds: 3600,
            },
            postgres: PostgresConfig {
                url: "postgresql://localhost:5432/milvuso".to_string(),
                max_connections: 10,
            },
            recommendation: RecommendationConfig {
                embedding_dim: 128,
                top_k: 50,
                similarity_threshold: 0.7,
                user_profile_update_interval: 300,
            },
            training: TrainingConfig {
                batch_size: 1024,
                learning_rate: 0.001,
                epochs: 10,
                model_save_interval: 3600,
                negative_sampling_ratio: 4.0,
            },
        }
    }
}

impl Config {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let settings = config::Config::builder()
            .add_source(config::File::with_name(path))
            .add_source(config::Environment::with_prefix("MILVUSO"))
            .build()?;
        
        Ok(settings.try_deserialize()?)
    }
}
