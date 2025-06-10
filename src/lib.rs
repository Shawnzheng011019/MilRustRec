pub mod config;
pub mod models;
pub mod services;
pub mod algorithms;
pub mod utils;

pub use config::Config;
pub use models::*;

use anyhow::Result;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub vector_db: Arc<services::vector_db::VectorDbService>,
    pub kafka_producer: Arc<services::kafka::KafkaProducer>,
    pub kafka_consumer: Arc<services::kafka::KafkaConsumer>,
    pub recommendation_service: Arc<services::recommendation::RecommendationService>,
    pub training_service: Arc<services::training::TrainingService>,
    pub redis_client: Arc<redis::Client>,
}

impl AppState {
    pub async fn new(config: Config) -> Result<Self> {
        let config = Arc::new(config);
        
        let vector_db = Arc::new(
            services::vector_db::VectorDbService::new(&config).await?
        );
        
        let kafka_producer = Arc::new(
            services::kafka::KafkaProducer::new(&config)?
        );
        
        let kafka_consumer = Arc::new(
            services::kafka::KafkaConsumer::new(&config)?
        );
        
        let redis_client = Arc::new(
            redis::Client::open(config.redis.url.as_str())?
        );
        
        let recommendation_service = Arc::new(
            services::recommendation::RecommendationService::new(
                vector_db.clone(),
                redis_client.clone(),
                config.clone(),
            ).await?
        );
        
        let training_service = Arc::new(
            services::training::TrainingService::new(
                vector_db.clone(),
                kafka_producer.clone(),
                config.clone(),
            ).await?
        );
        
        Ok(Self {
            config,
            vector_db,
            kafka_producer,
            kafka_consumer,
            recommendation_service,
            training_service,
            redis_client,
        })
    }
}

pub async fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
}
